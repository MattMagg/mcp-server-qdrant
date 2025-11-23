import argparse
import sys


def main():
    """
    Main entry point for the mcp-server-qdrant script defined
    in pyproject.toml. It runs the MCP server with a specific transport
    protocol.
    """
    print("Starting main()", file=sys.stderr, flush=True)

    # Parse the command-line arguments to determine the transport protocol.
    parser = argparse.ArgumentParser(description="mcp-server-qdrant")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
    )
    args = parser.parse_args()
    print(f"Args parsed: transport={args.transport}", file=sys.stderr, flush=True)

    # Import is done here to make sure environment variables are loaded
    # only after we make the changes.
    from mcp_server_qdrant.server import mcp
    import os
    print("Imports complete", file=sys.stderr, flush=True)

    # Map streamable-http to http (FastMCP's actual transport name)
    transport = "http" if args.transport == "streamable-http" else args.transport
    print(f"Using transport: {transport}", file=sys.stderr, flush=True)

    # For HTTP transports, get host and port from environment or use defaults
    if transport in ("http", "sse"):
        host = os.getenv("FASTMCP_HOST", "0.0.0.0")
        port = int(os.getenv("FASTMCP_PORT", "8000"))
        print(f"Starting HTTP server on {host}:{port}", file=sys.stderr, flush=True)
        mcp.run(transport=transport, host=host, port=port)
        print("AFTER mcp.run() - should not see this", file=sys.stderr, flush=True)
    else:
        print(f"Starting with transport: {transport}", file=sys.stderr, flush=True)
        mcp.run(transport=transport)
        print("AFTER mcp.run() - should not see this", file=sys.stderr, flush=True)
