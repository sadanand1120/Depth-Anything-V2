import time
import uuid
import base64
import numpy as np
import os
import asyncio
import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends, Header, Request
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
async def get_pointcloud(request: Request, depth_request: DepthRequest, _: bool = Depends(verify_api_key)):
    if not depth_request.max_depth:
        raise HTTPException(status_code=400, detail="max_depth is required for pointcloud")
    
    try:
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
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/metric_depth", response_model=DepthResponse)
async def get_metric_depth(request: Request, depth_request: DepthRequest, _: bool = Depends(verify_api_key)):
    if not depth_request.max_depth:
        raise HTTPException(status_code=400, detail="max_depth is required for metric depth")
    
    try:
        # Run prediction in thread pool to avoid blocking
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            depth_service.predict_metric_depth,
            depth_request.image,
            depth_request.image_url,
            depth_request.camera_intrinsics,
            depth_request.encoder,
            depth_request.dataset,
            depth_request.model_input_size,
            depth_request.max_depth
        )
        return DepthResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rel_depth", response_model=DepthResponse)
async def get_relative_depth(request: Request, depth_request: DepthRequest, _: bool = Depends(verify_api_key)):
    try:
        # Run prediction in thread pool to avoid blocking
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            depth_service.predict_relative_depth,
            depth_request.image,
            depth_request.image_url,
            depth_request.camera_intrinsics,
            depth_request.encoder,
            depth_request.dataset,
            depth_request.model_input_size
        )
        return DepthResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 