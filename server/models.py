from pydantic import BaseModel, Field
from typing import Optional


class CameraIntrinsics(BaseModel):
    fx: float = Field(..., description="Focal length x")
    fy: float = Field(..., description="Focal length y")
    cx: float = Field(..., description="Principal point x")
    cy: float = Field(..., description="Principal point y")


class DepthRequest(BaseModel):
    image: Optional[str] = Field(None, description="Base64 encoded image")
    image_url: Optional[str] = Field(None, description="URL to image")
    camera_intrinsics: Optional[CameraIntrinsics] = Field(None, description="Camera intrinsics (required for pointcloud)")
    encoder: str = Field(default="vitl", description="Model encoder")
    dataset: str = Field(default="hypersim", description="Dataset type")
    model_input_size: int = Field(default=518, description="Model input size")
    max_depth: Optional[float] = Field(None, description="Max depth for scaling (not needed for rel_depth)")


class PointCloudResponse(BaseModel):
    pointcloud: str = Field(..., description="Base64 encoded point cloud")
    pointcloud_shape: list = Field(..., description="Point cloud shape")
    depth_map: str = Field(..., description="Base64 encoded depth map")
    depth_shape: list = Field(..., description="Depth map shape")
    min: float = Field(..., description="Min depth value")
    max: float = Field(..., description="Max depth value")


class DepthResponse(BaseModel):
    depth_map: str = Field(..., description="Base64 encoded depth map")
    shape: list = Field(..., description="Depth map shape")
    min: float = Field(..., description="Min depth value")
    max: float = Field(..., description="Max depth value")


class HealthResponse(BaseModel):
    status: str
    device: str
    worker_id: int = Field(..., description="Worker ID")
    gpu_id: Optional[int] = Field(None, description="GPU ID (None if using CPU)")
    gpu_count: int = Field(..., description="Number of available GPUs")
    models_loaded: list = Field(..., description="List of model keys loaded")
    total_model_instances: int = Field(..., description="Total model instances across all GPUs")
    request_counter: int = Field(..., description="Total requests processed (for round-robin tracking)") 