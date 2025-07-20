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
    parser.add_argument("--reload", action="store_true", 
                       help="Enable auto-reload for development")
    
    # Authentication
    parser.add_argument("--api-key", type=str, 
                       help="API key for authentication (if not provided, auth is disabled)")
    
    # Logging
    parser.add_argument("--log-level", type=str, default="info", 
                       choices=["debug", "info", "warning", "error"],
                       help="Log level (default: info)")
    
    # Performance
    parser.add_argument("--max-requests", type=int, default=1000,
                       help="Maximum requests per worker before restart (default: 1000)")
    parser.add_argument("--max-requests-jitter", type=int, default=100,
                       help="Jitter for max requests (default: 100)")
    
    args = parser.parse_args()
    
    # Configure API key authentication
    if args.api_key:
        configure_auth(api_key=args.api_key, require_auth=True)
        print("✅ API key configured - authentication required")
    else:
        configure_auth(api_key=None, require_auth=False)
        print("ℹ️  No API key - authentication disabled")
    
    # Print configuration
    print(f"🚀 Starting Depth Anything V2 API Server")
    print(f"📍 Host: {args.host}")
    print(f"🔌 Port: {args.port}")
    print(f"👥 Workers: {args.workers}")
    print(f"📝 Log Level: {args.log_level}")
    print(f"🔄 Auto-reload: {'✅ Enabled' if args.reload else '❌ Disabled'}")
    print(f"📊 Max requests per worker: {args.max_requests}")
    print(f"🌐 API docs: http://{args.host}:{args.port}/docs")
    print(f"❤️  Health check: http://{args.host}:{args.port}/health")
    
    # Start server
    uvicorn.run(
        app, 
        host=args.host, 
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,  # Workers not compatible with reload
        log_level=args.log_level,
        max_requests=args.max_requests,
        max_requests_jitter=args.max_requests_jitter
    )


if __name__ == "__main__":
    main() 