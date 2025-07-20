#!/usr/bin/env python3
"""Start Depth Anything V2 API server"""

import argparse
import uvicorn
import sys
import os

# Import from server directory
from server.api_server import app, API_KEY, REQUIRE_AUTH


def main():
    parser = argparse.ArgumentParser(description="Depth Anything V2 API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--api-key", type=str, help="API key for authentication")
    
    args = parser.parse_args()
    
    # Configure API key authentication
    if args.api_key:
        app.API_KEY = args.api_key
        app.REQUIRE_AUTH = True
        print("✅ API key configured - authentication required")
    else:
        app.API_KEY = None
        app.REQUIRE_AUTH = False
        print("ℹ️  No API key - authentication disabled")
    
    print(f"Starting server on {args.host}:{args.port}")
    print(f"API docs: http://{args.host}:{args.port}/docs")
    
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main() 