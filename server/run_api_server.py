#!/usr/bin/env python3
"""Start Depth Anything V2 API server"""

import argparse
import uvicorn

# Import from server directory
from server.api_server import app, configure_auth


def main():
    parser = argparse.ArgumentParser(description="Depth Anything V2 API Server")
    
    # Server configuration
    parser.add_argument("--host", type=str, default="0.0.0.0", 
                       help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, 
                       help="Port to bind (default: 8000)")
    parser.add_argument("--workers", type=int, default=1, 
                       help="Number of worker processes (default: 1)")
    parser.add_argument("--max-concurrent-requests", type=int, default=4,
                       help="Maximum concurrent requests per worker (default: 4)")
    
    # Authentication
    parser.add_argument("--api-key", type=str, 
                       help="API key for authentication (if not provided, auth is disabled)")
    
    # Logging
    parser.add_argument("--log-level", type=str, default="info", 
                       choices=["debug", "info", "warning", "error"],
                       help="Log level (default: info)")
    
    args = parser.parse_args()
    
    # Configure API key authentication
    if args.api_key:
        # Set environment variables for multi-worker processes
        import os
        os.environ['DEPTHSERVER_API_KEY'] = args.api_key
        os.environ['DEPTHSERVER_REQUIRE_AUTH'] = 'true'
        os.environ['DEPTHSERVER_MAX_CONCURRENT_REQUESTS'] = str(args.max_concurrent_requests)
        
        # Also configure for main process
        configure_auth(api_key=args.api_key, require_auth=True)
        print("✅ API key configured - authentication required")
    else:
        # Clear environment variables
        import os
        os.environ.pop('DEPTHSERVER_API_KEY', None)
        os.environ['DEPTHSERVER_REQUIRE_AUTH'] = 'false'
        os.environ['DEPTHSERVER_MAX_CONCURRENT_REQUESTS'] = str(args.max_concurrent_requests)
        
        # Also configure for main process
        configure_auth(api_key=None, require_auth=False)
        print("ℹ️  No API key - authentication disabled")
    
    # Print configuration
    print(f"🚀 Starting Depth Anything V2 API Server")
    print(f"📍 Host: {args.host}")
    print(f"🔌 Port: {args.port}")
    print(f"👥 Workers: {args.workers}")
    print(f"🔒 Max Concurrent Requests: {args.max_concurrent_requests}")
    print(f"📝 Log Level: {args.log_level}")
    print(f"🌐 API docs: http://{args.host}:{args.port}/docs")
    print(f"❤️  Health check: http://{args.host}:{args.port}/health")
    
    # Start server
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