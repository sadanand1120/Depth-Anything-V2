import time
import uuid
import base64
import numpy as np
import os
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn

from server.models import DepthRequest, PointCloudResponse, DepthResponse, HealthResponse
from server.depth_service import depth_service

app = FastAPI(title="Depth Anything V2 API", version="1.0.0")

# Security scheme for API key authentication
security = HTTPBearer(auto_error=False)

# Global variables for API key configuration
API_KEY = None
REQUIRE_AUTH = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
async def get_pointcloud(request: DepthRequest, _: bool = Depends(verify_api_key)):
    if not request.max_depth:
        raise HTTPException(status_code=400, detail="max_depth is required for pointcloud")
    
    try:
        result = depth_service.predict_pointcloud(
            image=request.image,
            image_url=request.image_url,
            camera_intrinsics=request.camera_intrinsics,
            encoder=request.encoder,
            dataset=request.dataset,
            model_input_size=request.model_input_size,
            max_depth=request.max_depth
        )
        return PointCloudResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/metric_depth", response_model=DepthResponse)
async def get_metric_depth(request: DepthRequest, _: bool = Depends(verify_api_key)):
    if not request.max_depth:
        raise HTTPException(status_code=400, detail="max_depth is required for metric depth")
    
    try:
        result = depth_service.predict_metric_depth(
            image=request.image,
            image_url=request.image_url,
            camera_intrinsics=request.camera_intrinsics,
            encoder=request.encoder,
            dataset=request.dataset,
            model_input_size=request.model_input_size,
            max_depth=request.max_depth
        )
        return DepthResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rel_depth", response_model=DepthResponse)
async def get_relative_depth(request: DepthRequest, _: bool = Depends(verify_api_key)):
    try:
        result = depth_service.predict_relative_depth(
            image=request.image,
            image_url=request.image_url,
            camera_intrinsics=request.camera_intrinsics,
            encoder=request.encoder,
            dataset=request.dataset,
            model_input_size=request.model_input_size
        )
        return DepthResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 