# ai-assistant-client

A streaming chat client for Claude with first-class MCP tool support and
**progressive tool discovery** — the model searches the catalog and loads
schemas on demand instead of receiving every tool's schema on every turn.

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

## Install

```bash
pip install git+https://github.com/ryan-evans-git/ai-assistant-client.git
```

Or for local dev:

```bash
git clone https://github.com/ryan-evans-git/ai-assistant-client.git
cd ai-assistant-client
pip install -e ".[dev]"
```

Python 3.11+ required.

## Quickstart

```bash
export ANTHROPIC_API_KEY=sk-ant-...

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
| `ANTHROPIC_API_KEY` | _(required)_ | Claude credential. |
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
from anthropic import AsyncAnthropic
from ai_assistant_client import McpPool, McpServerConfig, ProgressiveToolRegistry
from ai_assistant_client.discovery import RemoteToolDescriptor
from ai_assistant_client.agent import AgentRunConfig, run_agent


async def main() -> None:
    configs = [
        McpServerConfig(name="oapi", sse_url="http://localhost:8765/sse"),
    ]
    async with McpPool(configs) as pool:
        tools = await pool.list_all_tools()
        registry = ProgressiveToolRegistry(
            [RemoteToolDescriptor(name=t.name, description=t.description, input_schema=t.input_schema) for t in tools]
        )
        client = AsyncAnthropic()
        history: list[dict] = []
        async for event in run_agent(
            user_message="What pet operations are available?",
            history=history,
            registry=registry,
            dispatcher=pool.call_tool,
            anthropic_client=client,
            config=AgentRunConfig(),
        ):
            print(event.type, event.data)


asyncio.run(main())
```

## Project layout

```
ai_assistant_client/
  app.py             # Starlette + SSE service
  agent.py           # Streaming Claude loop with tool dispatch
  discovery.py       # ProgressiveToolRegistry + meta-tools
  mcp_pool.py        # Live sessions to N upstream MCP servers

tests/               # pytest, no real Anthropic / MCP traffic
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
