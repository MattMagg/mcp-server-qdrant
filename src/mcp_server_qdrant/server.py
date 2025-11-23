import os
from fastmcp.server.middleware.authentication import AuthenticationMiddleware

from mcp_server_qdrant.mcp_server import QdrantMCPServer
from mcp_server_qdrant.settings import (
    EmbeddingProviderSettings,
    QdrantSettings,
    ToolSettings,
)

mcp = QdrantMCPServer(
    tool_settings=ToolSettings(),
    qdrant_settings=QdrantSettings(),
    embedding_provider_settings=EmbeddingProviderSettings(),
)

# Add authentication middleware if token is configured
auth_token = os.getenv("MCP_AUTH_TOKEN")
if auth_token:
    mcp.add_middleware(AuthenticationMiddleware(auth_token))
