"""
vexis-langchain — VEXIS Governance Adapter for LangChain
=========================================================

Automatic AI governance for every LLM call in your LangChain pipeline.
Just add the callback handler — prompts get checked, PII gets masked,
everything gets an audit trail.

Usage::

    from vexis_langchain import VexisCallbackHandler

    handler = VexisCallbackHandler(api_key="gp_live_xxx")

    # Works with any LangChain component
    chain.invoke({"input": "..."}, config={"callbacks": [handler]})

    # Or set globally
    from langchain_core.globals import set_llm_cache
    llm = ChatOpenAI(callbacks=[handler])
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

from vexis import Vexis, VerifyRequest, VerifyResponse, VexisError

__version__ = "0.1.0"
__all__ = ["VexisCallbackHandler", "VexisGovernanceError"]

logger = logging.getLogger("vexis_langchain")


# ── Errors ───────────────────────────────────────────────────


class VexisGovernanceError(Exception):
    """Raised when VEXIS blocks an LLM call or tool execution."""

    def __init__(self, message: str, decision: str, trace_id: str, reason: str):
        super().__init__(message)
        self.decision = decision
        self.trace_id = trace_id
        self.reason = reason


# ── Governance Result ────────────────────────────────────────


@dataclass(frozen=True)
class GovernanceRecord:
    """Record of a governance check, stored per run_id."""

    run_id: str
    event: str  # "llm_start", "chat_model_start", "tool_start", "chain_start"
    prompt: str
    decision: str
    reason: str
    trace_id: str
    findings_count: int
    latency_ms: float
    blocked: bool
    modified: bool
    sanitized_output: Optional[str] = None


# ── Callback Handler ─────────────────────────────────────────


class VexisCallbackHandler(BaseCallbackHandler):
    """
    LangChain callback handler that routes every LLM call and tool execution
    through VEXIS governance policies.

    Intercepts:
    - ``on_llm_start`` / ``on_chat_model_start`` — checks prompts before LLM inference
    - ``on_tool_start`` — checks tool inputs before execution
    - ``on_llm_end`` — optionally checks LLM outputs

    Behavior on BLOCKED:
    - If ``raise_on_block=True`` (default): raises ``VexisGovernanceError``
    - If ``raise_on_block=False``: logs the block and allows the call to proceed

    Behavior on MODIFIED:
    - The sanitized output is logged. Note that LangChain callbacks cannot modify
      the actual prompt in-flight — for full enforcement, use the VEXIS Gateway Proxy.

    Args:
        api_key: VEXIS project or agent API key.
        base_url: Gateway URL (default: https://gateway.vexis.io).
        check_prompts: Check prompts before LLM calls (default: True).
        check_outputs: Check LLM outputs after generation (default: False).
        check_tools: Check tool inputs before execution (default: True).
        raise_on_block: Raise VexisGovernanceError on BLOCKED (default: True).
        fail_open: Allow calls when gateway is unreachable (default: False).
        metadata: Extra metadata attached to every governance trace.

    Example::

        from vexis_langchain import VexisCallbackHandler
        from langchain_openai import ChatOpenAI

        handler = VexisCallbackHandler(api_key="gp_live_xxx")
        llm = ChatOpenAI(model="gpt-4o", callbacks=[handler])

        # Every call is now governed
        result = llm.invoke("Transfer $50,000 to account DE89...")
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://gateway.vexis.io",
        check_prompts: bool = True,
        check_outputs: bool = False,
        check_tools: bool = True,
        raise_on_block: bool = True,
        fail_open: bool = False,
        metadata: Optional[dict[str, Any]] = None,
    ):
        super().__init__()
        self._client = Vexis(api_key=api_key, base_url=base_url)
        self._check_prompts = check_prompts
        self._check_outputs = check_outputs
        self._check_tools = check_tools
        self._raise_on_block = raise_on_block
        self._fail_open = fail_open
        self._base_metadata = metadata or {}
        self._records: list[GovernanceRecord] = []

    # ── LangChain Callback Interface ─────────────────────────

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Check prompts before they reach the LLM."""
        if not self._check_prompts:
            return

        model_name = serialized.get("id", ["unknown"])[-1] if serialized.get("id") else "unknown"

        for prompt in prompts:
            self._check(
                text=prompt,
                event="llm_start",
                run_id=str(run_id),
                extra_metadata={
                    "model": model_name,
                    "langchain_event": "on_llm_start",
                    **(metadata or {}),
                },
            )

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Check chat messages before they reach the chat model."""
        if not self._check_prompts:
            return

        model_name = serialized.get("id", ["unknown"])[-1] if serialized.get("id") else "unknown"

        for message_list in messages:
            # Concatenate all messages into a single governance check
            combined = "\n".join(
                f"[{msg.type}] {msg.content}" for msg in message_list if isinstance(msg.content, str)
            )
            if combined.strip():
                self._check(
                    text=combined,
                    event="chat_model_start",
                    run_id=str(run_id),
                    extra_metadata={
                        "model": model_name,
                        "message_count": len(message_list),
                        "langchain_event": "on_chat_model_start",
                        **(metadata or {}),
                    },
                )

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Check tool inputs before execution."""
        if not self._check_tools:
            return

        tool_name = serialized.get("name", "unknown_tool")

        self._check(
            text=f"[Tool: {tool_name}] {input_str}",
            event="tool_start",
            run_id=str(run_id),
            extra_metadata={
                "tool": tool_name,
                "langchain_event": "on_tool_start",
                **(metadata or {}),
            },
        )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Optionally check LLM outputs after generation."""
        if not self._check_outputs:
            return

        for generation_list in response.generations:
            for generation in generation_list:
                if generation.text:
                    self._check(
                        text=generation.text,
                        event="llm_end",
                        run_id=str(run_id),
                        extra_metadata={"langchain_event": "on_llm_end"},
                    )

    # ── Governance Records Access ────────────────────────────

    @property
    def records(self) -> list[GovernanceRecord]:
        """All governance records from this handler's lifetime."""
        return list(self._records)

    @property
    def blocked_count(self) -> int:
        """Number of blocked requests."""
        return sum(1 for r in self._records if r.blocked)

    @property
    def trace_ids(self) -> list[str]:
        """All trace IDs for audit reference."""
        return [r.trace_id for r in self._records if r.trace_id]

    def clear_records(self) -> None:
        """Clear stored governance records."""
        self._records.clear()

    # ── Internal ─────────────────────────────────────────────

    def _check(
        self,
        text: str,
        event: str,
        run_id: str,
        extra_metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Run a governance check against VEXIS."""
        merged_metadata = {
            **self._base_metadata,
            "source": "langchain",
            "run_id": run_id,
            **(extra_metadata or {}),
        }

        try:
            result = self._client.verify(
                VerifyRequest(
                    prompt=text,
                    metadata=merged_metadata,
                )
            )

            record = GovernanceRecord(
                run_id=run_id,
                event=event,
                prompt=text[:200] + "..." if len(text) > 200 else text,
                decision=result.decision.value,
                reason=result.reason,
                trace_id=result.trace_id,
                findings_count=len(result.findings),
                latency_ms=result.latency_ms,
                blocked=result.is_blocked,
                modified=result.decision.value == "MODIFIED",
                sanitized_output=result.output if result.decision.value == "MODIFIED" else None,
            )
            self._records.append(record)

            if result.is_blocked:
                logger.warning(
                    "🚫 VEXIS BLOCKED [%s] %s — %s (trace: %s)",
                    event, text[:80], result.reason, result.trace_id,
                )
                if self._raise_on_block:
                    raise VexisGovernanceError(
                        f"Blocked by VEXIS governance: {result.reason}",
                        decision="BLOCKED",
                        trace_id=result.trace_id,
                        reason=result.reason,
                    )

            elif result.decision.value == "MODIFIED":
                logger.info(
                    "✏️ VEXIS MODIFIED [%s] — PII redacted (trace: %s)",
                    event, result.trace_id,
                )

            else:
                logger.debug(
                    "✅ VEXIS ALLOWED [%s] (trace: %s, %dms)",
                    event, result.trace_id, result.latency_ms,
                )

        except VexisGovernanceError:
            raise  # Re-raise governance blocks

        except VexisError as e:
            if self._fail_open:
                logger.warning("⚠️ VEXIS gateway error, fail-open: %s", e)
            else:
                logger.error("🚫 VEXIS gateway error, fail-closed: %s", e)
                raise VexisGovernanceError(
                    f"Gateway error (fail-closed): {e}",
                    decision="ERROR",
                    trace_id="",
                    reason=str(e),
                )

        except Exception as e:
            if self._fail_open:
                logger.warning("⚠️ Unexpected error in VEXIS check, fail-open: %s", e)
            else:
                logger.error("🚫 Unexpected error in VEXIS check, fail-closed: %s", e)
                raise
