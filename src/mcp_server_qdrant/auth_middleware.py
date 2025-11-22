"""Authentication middleware for MCP server."""
import os
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


class BearerAuthMiddleware(BaseHTTPMiddleware):
    """Middleware to validate Bearer token authentication."""

    async def dispatch(self, request, call_next):
        # Get expected token from environment
        expected_token = os.getenv("MCP_AUTH_TOKEN")

        # If no token configured, allow all requests (backwards compatible)
        if not expected_token:
            return await call_next(request)

        # Check Authorization header
        auth_header = request.headers.get("Authorization", "")

        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                {"error": "Missing or invalid Authorization header"},
                status_code=401
            )

        provided_token = auth_header.replace("Bearer ", "", 1)

        if provided_token != expected_token:
            return JSONResponse(
                {"error": "Invalid authentication token"},
                status_code=401
            )

        # Token is valid, proceed with request
        return await call_next(request)
