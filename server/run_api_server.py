#!/usr/bin/env python3
"""Start Depth Anything V2 API server"""

import argparse
import uvicorn
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app import app


def main():
    parser = argparse.ArgumentParser(description="Depth Anything V2 API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    print(f"Starting server on {args.host}:{args.port}")
    print(f"API docs: http://{args.host}:{args.port}/docs")
    
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main() 