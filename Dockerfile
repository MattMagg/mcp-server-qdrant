FROM python:3.11-slim

WORKDIR /app

# Install uv for package management
RUN pip install --no-cache-dir uv

# Copy local source code with optimal search modifications
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install from local source with all dependencies (voyageai, sentence-transformers, etc.)
RUN uv pip install --system --no-cache-dir .

# Expose the default port for HTTP transport
EXPOSE 8000

# Set environment variables with defaults that can be overridden at runtime
ENV FASTMCP_HOST="0.0.0.0"
ENV FASTMCP_PORT="8000"
ENV QDRANT_URL=""
ENV QDRANT_API_KEY=""
ENV COLLECTION_NAME="light-on-yoga"
ENV VOYAGE_API_KEY=""
ENV TOKENIZERS_PARALLELISM="true"

# Run the server with streamable HTTP transport
CMD ["python", "-m", "mcp_server_qdrant.main", "--transport", "streamable-http"]
