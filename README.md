# vexis-langchain

VEXIS AI Governance adapter for **LangChain** — automatic policy checks, PII masking, and audit trails for every LLM call in your pipeline.

[![PyPI](https://img.shields.io/pypi/v/vexis-langchain.svg?style=flat-square)](https://pypi.org/project/vexis-langchain/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg?style=flat-square)](https://opensource.org/licenses/Apache-2.0)

---

Add one callback handler. Every prompt, every tool call, every LLM output gets checked against your [VEXIS](https://vexis.io) governance policies. PII is detected. Secrets are caught. Blocked requests raise exceptions before they reach the model.

## Installation

```bash
pip install vexis-langchain
```

## Quick Start

```python
from vexis_langchain import VexisCallbackHandler
from langchain_openai import ChatOpenAI

# Create the governance handler
handler = VexisCallbackHandler(api_key="gp_live_xxx")

# Attach to any LangChain component
llm = ChatOpenAI(model="gpt-4o", callbacks=[handler])

# Every call is now governed
result = llm.invoke("Transfer $50,000 to account DE89370400440532013000")
# → VexisGovernanceError: Blocked — PII detected (IBAN)
```

## What Gets Checked

| Event | When | What |
|-------|------|------|
| `on_llm_start` | Before text completion | Prompt text |
| `on_chat_model_start` | Before chat model call | All messages concatenated |
| `on_tool_start` | Before tool execution | Tool name + input |
| `on_llm_end` | After generation (optional) | LLM output text |

## Configuration

```python
handler = VexisCallbackHandler(
    api_key="gp_live_xxx",
    base_url="https://gateway.vexis.io",  # On-prem endpoint
    check_prompts=True,       # Check inputs before LLM (default: True)
    check_outputs=False,      # Check LLM outputs (default: False)
    check_tools=True,         # Check tool inputs (default: True)
    raise_on_block=True,      # Raise exception on BLOCKED (default: True)
    fail_open=False,          # Allow calls when gateway down (default: False)
    metadata={"team": "ml"},  # Extra metadata on every trace
)
```

## Behavior on Decisions

| Decision | `raise_on_block=True` | `raise_on_block=False` |
|----------|----------------------|----------------------|
| `ALLOWED` | Call proceeds | Call proceeds |
| `MODIFIED` | Call proceeds, PII redaction logged | Call proceeds, PII redaction logged |
| `BLOCKED` | Raises `VexisGovernanceError` | Logs warning, call proceeds |
| `ERROR` | Depends on `fail_open` | Depends on `fail_open` |

**Note:** LangChain callbacks are observational — they cannot modify prompts in-flight. For full input rewriting (PII masking before the LLM sees it), use the [VEXIS Gateway Proxy](https://docs.vexis.io/guides/gateway-proxy).

## Governance Records

Access the audit trail programmatically:

```python
# After running your chain
print(f"Blocked: {handler.blocked_count}")
print(f"Trace IDs: {handler.trace_ids}")

for record in handler.records:
    print(f"{record.event}: {record.decision} ({record.latency_ms:.0f}ms) — {record.trace_id}")
```

## Error Handling

```python
from vexis_langchain import VexisCallbackHandler, VexisGovernanceError

handler = VexisCallbackHandler(api_key="gp_live_xxx")

try:
    result = llm.invoke("Send SSN 123-45-6789 to the client", config={"callbacks": [handler]})
except VexisGovernanceError as e:
    print(e.decision)   # "BLOCKED"
    print(e.trace_id)   # "trc_abc123"
    print(e.reason)     # "PII detected: Social Security Number"
```

## With Agents

```python
from langchain.agents import create_react_agent

handler = VexisCallbackHandler(api_key="gp_live_xxx", check_tools=True)

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, callbacks=[handler])

# Both LLM calls AND tool executions are governed
result = agent_executor.invoke({"input": "Delete all customer records"})
```

## With Chains (LCEL)

```python
from langchain_core.prompts import ChatPromptTemplate

handler = VexisCallbackHandler(api_key="gp_live_xxx")

chain = ChatPromptTemplate.from_template("Summarize: {text}") | llm
result = chain.invoke({"text": sensitive_doc}, config={"callbacks": [handler]})
```

## Requirements

- Python 3.9+
- `vexis-sdk` >= 0.5.0
- `langchain-core` >= 0.2.0

## Links

- [Documentation](https://docs.vexis.io/integrations/langchain)
- [VEXIS Dashboard](https://app.vexis.io)
- [vexis-sdk (Python)](https://pypi.org/project/vexis-sdk)
- [GitHub](https://github.com/disruptivetrends/vexis-langchain)

## License

[Apache 2.0](./LICENSE)
