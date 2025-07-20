#!/usr/bin/env python3

import argparse
import uvicorn

from server.api_server import app


def main():
    parser = argparse.ArgumentParser(description="Depth Anything V2 API Server")
    
    parser.add_argument("--host", type=str, default="0.0.0.0", 
                       help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, 
                       help="Port to bind")
    parser.add_argument("--workers", type=int, default=4, 
                       help="Number of worker processes, per gpu")
    parser.add_argument("--log-level", type=str, default="info", 
                       choices=["debug", "info", "warning", "error"],
                       help="Log level")
    
    args = parser.parse_args()
    
    if args.workers > 1:
        # For multiple workers, use import string
        uvicorn.run(
            "server.api_server:app",
            host=args.host,
            port=args.port,
            workers=args.workers,
            log_level=args.log_level
        )
    else:
        # For single worker, use app object directly
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level=args.log_level
        )


if __name__ == "__main__":
    main() 