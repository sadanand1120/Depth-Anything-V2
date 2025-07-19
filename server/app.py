import time
import uuid
import base64
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from models import DepthRequest, PointCloudResponse, DepthResponse, HealthResponse
from depth_service import depth_service

app = FastAPI(title="Depth Anything V2 API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
async def health():
    return depth_service.get_health()

@app.post("/pc", response_model=PointCloudResponse)
async def get_pointcloud(request: DepthRequest):
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
async def get_metric_depth(request: DepthRequest):
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
async def get_relative_depth(request: DepthRequest):
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