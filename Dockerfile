# syntax=docker/dockerfile:1.6
FROM python:3.14-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY pyproject.toml README.md /app/
RUN pip install --upgrade pip && pip install .

COPY ai_assistant_client /app/ai_assistant_client

ENV AI_ASSISTANT_CLIENT_HOST=0.0.0.0 \
    AI_ASSISTANT_CLIENT_PORT=8080

EXPOSE 8080

# Configure upstream MCP servers via the MCP_SERVERS env (JSON
# array of {name, command|sse_url, ...}).  Anthropic auth comes
# from ANTHROPIC_API_KEY.
ENTRYPOINT ["ai-assistant-client"]
