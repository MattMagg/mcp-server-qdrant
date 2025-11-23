import sys
import os
print("Starting server...", file=sys.stderr, flush=True)
os.environ.setdefault("QDRANT_URL", "https://test.com")
os.environ.setdefault("COLLECTION_NAME", "test")

from mcp_server_qdrant.server import mcp
print("MCP imported successfully", file=sys.stderr, flush=True)
print(f"MCP type: {type(mcp)}", file=sys.stderr, flush=True)
print("Calling mcp.run()...", file=sys.stderr, flush=True)
mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)
print("After mcp.run() - THIS SHOULD NOT PRINT IF SERVER IS BLOCKING", file=sys.stderr, flush=True)
