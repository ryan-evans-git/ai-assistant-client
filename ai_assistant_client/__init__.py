"""ai-assistant-client: streaming chat client for Claude + MCP."""

from ai_assistant_client.discovery import (
    ProgressiveToolRegistry,
    RemoteToolDescriptor,
)
from ai_assistant_client.mcp_pool import McpPool, McpServerConfig

__all__ = [
    "McpPool",
    "McpServerConfig",
    "ProgressiveToolRegistry",
    "RemoteToolDescriptor",
]
__version__ = "0.1.0"
