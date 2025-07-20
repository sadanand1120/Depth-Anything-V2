import time
import asyncio
import logging
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn

from server.models import DepthRequest, PointCloudResponse, DepthResponse, HealthResponse
from server.depth_service import DepthService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Depth Anything V2 API", version="1.0.0")

# Security scheme for API key authentication
security = HTTPBearer(auto_error=False)

# Global variables for API key configuration
API_KEY = None
REQUIRE_AUTH = False

# Create service instance
depth_service = DepthService()

# Rate limiting: limit concurrent requests to prevent GPU OOM
# Read from environment variable for multi-worker setup
import os
MAX_CONCURRENT_REQUESTS = int(os.environ.get('DEPTHSERVER_MAX_CONCURRENT_REQUESTS', '4'))
request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# Function to configure API key (called from run_api_server.py)
def configure_auth(api_key: str = None, require_auth: bool = False):
    global API_KEY, REQUIRE_AUTH
    API_KEY = api_key
    REQUIRE_AUTH = require_auth

# Configure auth from environment variables (for multi-worker setup)
def configure_auth_from_env():
    """Configure authentication from environment variables for multi-worker processes"""
    import os
    api_key = os.environ.get('DEPTHSERVER_API_KEY')
    require_auth = os.environ.get('DEPTHSERVER_REQUIRE_AUTH', 'false').lower() == 'true'
    if api_key or require_auth:
        configure_auth(api_key=api_key, require_auth=require_auth)

# Configure auth on module import (for multi-worker processes)
configure_auth_from_env()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """VLLM-style request logging middleware"""
    start_time = time.time()
    
    # Get client IP
    client_ip = request.client.host if request.client else "unknown"
    
    # Log request start
    logger.info(f"{client_ip} - \"{request.method} {request.url.path} HTTP/1.1\" - Processing")
    
    # Process request
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time
    
    # Log response
    logger.info(f"{client_ip} - \"{request.method} {request.url.path} HTTP/1.1\" {response.status_code} - {process_time:.3f}s")
    
    return response

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key from Authorization header"""
    if not REQUIRE_AUTH:
        return True
    
    if not API_KEY:
        # If no API key is set but auth is required, deny all requests
        raise HTTPException(status_code=401, detail="API key required but not configured")
    
    if not credentials:
        raise HTTPException(status_code=401, detail="API key required")
    
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return True

@app.get("/health", response_model=HealthResponse)
async def health():
    return depth_service.get_health()

@app.post("/pc", response_model=PointCloudResponse)
async def get_pointcloud(depth_request: DepthRequest, _: bool = Depends(verify_api_key)):
    if not depth_request.max_depth:
        raise HTTPException(status_code=400, detail="max_depth is required for pointcloud")
    
    if not depth_request.camera_intrinsics:
        raise HTTPException(status_code=400, detail="camera_intrinsics is required for pointcloud")
    
    try:
        async with request_semaphore:
            # Run prediction in thread pool to avoid blocking
            result = await asyncio.get_event_loop().run_in_executor(
                None, 
                depth_service.predict_pointcloud,
                depth_request.image,
                depth_request.image_url,
                depth_request.camera_intrinsics,
                depth_request.encoder,
                depth_request.dataset,
                depth_request.model_input_size,
                depth_request.max_depth
            )
        return PointCloudResponse(**result)
    except Exception as e:
        logger.error(f"Error in pointcloud endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/metric_depth", response_model=DepthResponse)
async def get_metric_depth(depth_request: DepthRequest, _: bool = Depends(verify_api_key)):
    if not depth_request.max_depth:
        raise HTTPException(status_code=400, detail="max_depth is required for metric depth")
    
    try:
        async with request_semaphore:
            # Run prediction in thread pool to avoid blocking
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                depth_service.predict_metric_depth,
                depth_request.image,
                depth_request.image_url,
                depth_request.encoder,
                depth_request.dataset,
                depth_request.model_input_size,
                depth_request.max_depth
            )
        return DepthResponse(**result)
    except Exception as e:
        logger.error(f"Error in metric_depth endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/rel_depth", response_model=DepthResponse)
async def get_relative_depth(depth_request: DepthRequest, _: bool = Depends(verify_api_key)):
    try:
        async with request_semaphore:
            # Run prediction in thread pool to avoid blocking
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                depth_service.predict_relative_depth,
                depth_request.image,
                depth_request.image_url,
                depth_request.encoder,
                depth_request.dataset,
                depth_request.model_input_size
            )
        return DepthResponse(**result)
    except Exception as e:
        logger.error(f"Error in relative_depth endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 