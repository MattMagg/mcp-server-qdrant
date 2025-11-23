import os
from fastmcp.server.auth import StaticTokenVerifier

from mcp_server_qdrant.mcp_server import QdrantMCPServer
from mcp_server_qdrant.settings import (
    EmbeddingProviderSettings,
    QdrantSettings,
    ToolSettings,
)

# Configure authentication if token is provided
auth_token = os.getenv("MCP_AUTH_TOKEN")
auth = StaticTokenVerifier(tokens={auth_token: {}}) if auth_token else None

mcp = QdrantMCPServer(
    tool_settings=ToolSettings(),
    qdrant_settings=QdrantSettings(),
    embedding_provider_settings=EmbeddingProviderSettings(),
    auth=auth,
)
