from pydantic import BaseModel, Field
from typing import Optional, Union
import base64


class CameraIntrinsics(BaseModel):
    fx: float = Field(..., description="Focal length x")
    fy: float = Field(..., description="Focal length y")
    cx: float = Field(..., description="Principal point x")
    cy: float = Field(..., description="Principal point y")


class DepthRequest(BaseModel):
    image: Optional[str] = Field(None, description="Base64 encoded image")
    image_url: Optional[str] = Field(None, description="URL to image")
    camera_intrinsics: CameraIntrinsics
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
    gpu_count: int = Field(..., description="Number of available GPUs")
    models_loaded: list 