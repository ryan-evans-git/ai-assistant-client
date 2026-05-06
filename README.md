# ai-assistant-client

A provider-agnostic streaming chat client with first-class MCP tool
support and **progressive tool discovery** — the model searches the
catalog and loads schemas on demand instead of receiving every tool's
schema on every turn.

Pick your LLM with a single env var: **Anthropic Claude**, **OpenAI**,
**Google Gemini**, or **AWS Bedrock (Converse)**.

```
client/v1
├── POST /chat        → SSE stream of agent events
├── GET  /tools       → upstream MCP catalog dump
└── GET  /healthz
```

## Why progressive discovery

A naive MCP integration loads every upstream tool's schema into the system
prompt. With ~50 tools that's tens of thousands of tokens spent on every
turn — even when only one tool is relevant.

This client exposes only **two** tools to Claude by default:

- `tool_search(query, max_results=5)` — top-N matches by name/description.
- `tool_load(names)` — promotes the named tools' full schemas into the
  conversation for the rest of the session.

The model's loop becomes: **search → load → invoke**. Token spend on tool
schemas scales with what the model actually needs, not with catalog size.
Loaded schemas persist for the session — once `tool_load`ed, a tool can
be called directly on subsequent turns.

## Pluggable LLM providers

Set `LLM_PROVIDER` to pick the backend; the matching SDK is the only
one that needs to be installed.

| `LLM_PROVIDER` | Default model              | Credential env                     | Install extra |
|----------------|----------------------------|------------------------------------|---------------|
| `anthropic`    | `claude-sonnet-4-6`        | `ANTHROPIC_API_KEY`                | `[anthropic]` |
| `openai`       | `gpt-4o`                   | `OPENAI_API_KEY`                   | `[openai]`    |
| `gemini`       | `gemini-2.0-flash`         | `GEMINI_API_KEY` / `GOOGLE_API_KEY`| `[gemini]`    |
| `bedrock`      | `anthropic.claude-3-5-sonnet-20241022-v2:0` | standard AWS credential chain | `[bedrock]` |

Override the model with `LLM_MODEL`. The progressive-tool-discovery
behavior, SSE event shape, and MCP integration are identical across
providers — translation lives entirely in the per-provider adapters
under `ai_assistant_client/llm/`.

## Why a custom provider abstraction (and not LiteLLM/aisuite)

A "universal LLM SDK" is the obvious shortcut. We didn't take it for
two reasons:

**1. Tool-call semantics are the part you don't want a third party
deciding for you.** With MCP, tool input schemas are user-authored
JSON Schema and round-trip back to remote servers as `tool_use_id` →
`tool_result` pairs. Every provider expresses this differently
(Anthropic blocks, OpenAI `tool_calls[]`, Gemini `function_call`
parts, Bedrock Converse `toolUse` blocks), and universal wrappers
normalize to whichever shape they were born from — usually OpenAI's,
which loses fidelity (no per-result error flags, weaker streaming
semantics for partial JSON, ad-hoc `id` correlation). Owning ~100
lines of adapter per provider keeps the loop, the registry, and the
SSE contract unchanged regardless of backend.

**2. Supply-chain risk.** This service holds production LLM
credentials and brokers tool calls into your MCP fleet — it's a
high-value target. Pulling in a broad LLM-gateway dependency widens
the attack surface considerably:

- Every provider SDK is also a transitive dep, even ones you don't use.
- The wrapper itself becomes a privileged credential consumer; a
  compromised release can exfiltrate `ANTHROPIC_API_KEY`,
  `OPENAI_API_KEY`, AWS keys, and request bodies in one shot.
- Some popular LLM-gateway packages ship telemetry, hosted-proxy, or
  auto-update features that quietly route requests through
  third-party infrastructure unless explicitly disabled.
- The wider Python AI tooling ecosystem has seen multiple recent
  supply-chain incidents (typo-squatted packages, credential-stealing
  releases, and a high-profile gateway maintainer key compromise) —
  treat any package that touches model credentials as
  security-critical.

By depending only on the first-party SDK for the provider you've
actually selected (lazy-imported, behind an extras marker), the
service holds no third-party code on the credential path. Anthropic,
OpenAI, Google, and AWS all have well-staffed security teams and
signed releases; that is a defensible perimeter. A wrapper sitting in
front of all four is not.

If you want LiteLLM later, you can write a fifth adapter in ~100
lines — but for the four major providers we ship, it isn't worth the
risk.

## Install

```bash
# Just the core + Anthropic.
pip install "git+https://github.com/ryan-evans-git/ai-assistant-client.git#egg=ai-assistant-client[anthropic]"

# Or any other single provider:
pip install "git+...#egg=ai-assistant-client[openai]"
pip install "git+...#egg=ai-assistant-client[gemini]"
pip install "git+...#egg=ai-assistant-client[bedrock]"

# Everything:
pip install "git+...#egg=ai-assistant-client[all]"
```

Local dev:

```bash
git clone https://github.com/ryan-evans-git/ai-assistant-client.git
cd ai-assistant-client
pip install -e ".[dev,all]"
```

Python 3.11+ required.

## Quickstart

```bash
# Pick a provider (default: anthropic).
export LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=sk-ant-...
# Optional: pin a specific model.
# export LLM_MODEL=claude-sonnet-4-6

# Connect to one MCP server over stdio + one over SSE.
export MCP_SERVERS='[
  {"name":"oapi","sse_url":"http://localhost:8765/sse"},
  {"name":"local","command":"ai-assistant-server","args":["--tools-dir","./tools"]}
]'

ai-assistant-client
```

Or via Docker:

```bash
docker build -t ai-assistant-client .
docker run -p 8080:8080 \
  -e LLM_PROVIDER -e LLM_MODEL \
  -e ANTHROPIC_API_KEY \
  -e MCP_SERVERS \
  ai-assistant-client
```

## Use the SSE chat endpoint

```bash
curl -N -X POST http://localhost:8080/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "List the available pet operations and find pet 1."}'
```

Event stream (each is a JSON `data:` payload):

| Event | Data |
|---|---|
| `user_message` | `{ "content": "..." }` |
| `text_delta` | `{ "text": "..." }` |
| `tool_use` | `{ "id", "name", "input" }` |
| `tool_result` | `{ "id", "content" }` |
| `tool_error` | `{ "id", "name", "error" }` |
| `turn_complete` | `{ "stop_reason", ... }` |

The client appends user / assistant / tool blocks to its in-memory
history during the stream. Persist `history` between turns by passing
the prior turns back in the next `POST /chat` body.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | One of `anthropic`, `openai`, `gemini`, `bedrock`. |
| `LLM_MODEL` | provider default | Override the model name passed to the provider. |
| `ANTHROPIC_API_KEY` | — | Required when `LLM_PROVIDER=anthropic`. |
| `OPENAI_API_KEY` | — | Required when `LLM_PROVIDER=openai`. |
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | — | Required when `LLM_PROVIDER=gemini`. |
| `AWS_*` (standard chain) | — | Used when `LLM_PROVIDER=bedrock`. |
| `MCP_SERVERS` | _(optional)_ | JSON array of upstream MCP servers. |
| `AI_ASSISTANT_CLIENT_HOST` | `127.0.0.1` | Bind host. |
| `AI_ASSISTANT_CLIENT_PORT` | `8080` | Bind port. |
| `AI_ASSISTANT_CLIENT_LOG_LEVEL` | `INFO` | Logger level. |

`MCP_SERVERS` entry shape:

```json
{
  "name": "oapi",
  "sse_url": "http://localhost:8765/sse",       // OR command/args
  "command": "ai-assistant-server",
  "args": ["--tools-dir", "./tools"],
  "env": {"ANY_EXTRA_ENV": "..."},
  "forwarded_credentials": {"bearerAuth": "<token>"}
}
```

`forwarded_credentials` is passed to the upstream MCP server keyed by
OpenAPI security-scheme name — useful for acting on behalf of an
authenticated end-user.

## Library use (no HTTP server)

```python
import asyncio
from ai_assistant_client import McpPool, McpServerConfig, ProgressiveToolRegistry
from ai_assistant_client.discovery import RemoteToolDescriptor
from ai_assistant_client.agent import AgentRunConfig, run_agent
from ai_assistant_client.llm import default_model, make_provider


async def main() -> None:
    configs = [
        McpServerConfig(name="oapi", sse_url="http://localhost:8765/sse"),
    ]
    async with McpPool(configs) as pool:
        tools = await pool.list_all_tools()
        registry = ProgressiveToolRegistry(
            [RemoteToolDescriptor(name=t.name, description=t.description, input_schema=t.input_schema) for t in tools]
        )
        provider = make_provider("anthropic")  # or "openai" / "gemini" / "bedrock"
        history: list[dict] = []
        async for event in run_agent(
            user_message="What pet operations are available?",
            history=history,
            registry=registry,
            dispatcher=pool.call_tool,
            provider=provider,
            config=AgentRunConfig(model=default_model("anthropic")),
        ):
            print(event.type, event.data)


asyncio.run(main())
```

## Project layout

```
ai_assistant_client/
  app.py                 # Starlette + SSE service
  agent.py               # Streaming agent loop with tool dispatch
  discovery.py           # ProgressiveToolRegistry + meta-tools
  mcp_pool.py            # Live sessions to N upstream MCP servers
  llm/
    base.py              # LLMProvider ABC + normalized event types
    anthropic_provider.py
    openai_provider.py
    gemini_provider.py
    bedrock_provider.py  # Bedrock Converse API
    __init__.py          # make_provider() factory + default_model()

tests/                   # pytest, no real LLM / MCP traffic
```

## Companion projects

- [ai-assistant-server](https://github.com/ryan-evans-git/ai-assistant-server)
  — generic MCP server that auto-loads tools from OpenAPI specs.
- [ai-assistant-ui](https://github.com/ryan-evans-git/ai-assistant-ui)
  — drop-in React chat panel that consumes this client's SSE events.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
