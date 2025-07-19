#!/usr/bin/env python3
"""
Main script to run the Depth Anything V2 API server.

Usage:
    python run_api_server.py [--host HOST] [--port PORT] [--reload]
"""

import argparse
import uvicorn
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import app


def main():
    parser = argparse.ArgumentParser(description="Run Depth Anything V2 API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", 
                       help="Host to bind the server to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, 
                       help="Port to bind the server to (default: 8000)")
    parser.add_argument("--reload", action="store_true", 
                       help="Enable auto-reload for development")
    parser.add_argument("--workers", type=int, default=1, 
                       help="Number of worker processes (default: 1)")
    
    args = parser.parse_args()
    
    print(f"Starting Depth Anything V2 API Server...")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Reload: {args.reload}")
    print(f"Workers: {args.workers}")
    print(f"API Documentation: http://{args.host}:{args.port}/docs")
    print(f"Health Check: http://{args.host}:{args.port}/v1/depth/health")
    print("-" * 50)
    
    try:
        uvicorn.run(
            "server.app:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers if not args.reload else 1,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 