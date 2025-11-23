import sys
print("Python starting...", file=sys.stderr)
from mcp_server_qdrant.server import mcp
print("MCP imported...", file=sys.stderr)
print(f"Running with streamable-http...", file=sys.stderr)
mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)
print("After mcp.run...", file=sys.stderr)
