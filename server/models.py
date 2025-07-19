from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from enum import Enum
import base64


class ResponseFormat(str, Enum):
    POINTCLOUD = "pointcloud"
    DEPTH_RELATIVE = "depth_relative"
    DEPTH_METRIC = "depth_metric"
    ALL = "all"


class CameraIntrinsics(BaseModel):
    fx: float = Field(..., description="Focal length in x direction")
    fy: float = Field(..., description="Focal length in y direction")
    cx: float = Field(..., description="Principal point x coordinate")
    cy: float = Field(..., description="Principal point y coordinate")


class DepthOptions(BaseModel):
    encoder: str = Field(default="vitl", description="Model encoder (vitl, vitb, vits)")
    dataset: str = Field(default="hypersim", description="Dataset type (hypersim, vkitti)")
    max_depth: float = Field(default=1.0, description="Maximum depth value")
    model_input_size: int = Field(default=518, description="Model input size")


class UserContent(BaseModel):
    image: str = Field(..., description="Base64 encoded image")
    camera_intrinsics: CameraIntrinsics = Field(..., description="Camera intrinsics")
    options: Optional[DepthOptions] = Field(default=None, description="Depth prediction options")


class Message(BaseModel):
    role: str = Field(..., description="Message role (user, system, assistant)")
    content: Union[str, UserContent] = Field(..., description="Message content")


class DepthRequest(BaseModel):
    model: str = Field(default="depth-anything-v2", description="Model name")
    messages: List[Message] = Field(..., description="List of messages")
    response_format: ResponseFormat = Field(default=ResponseFormat.ALL, description="Response format")
    temperature: float = Field(default=0.0, description="Temperature (not used for depth prediction)")


class DepthResponse(BaseModel):
    id: str = Field(..., description="Response ID")
    object: str = Field(default="chat.completion", description="Object type")
    created: int = Field(..., description="Timestamp")
    model: str = Field(..., description="Model used")
    choices: List[Dict[str, Any]] = Field(..., description="Response choices")
    usage: Dict[str, int] = Field(..., description="Usage statistics")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    device: str = Field(..., description="Device being used")


class ModelInfo(BaseModel):
    id: str = Field(..., description="Model ID")
    object: str = Field(default="model", description="Object type")
    created: int = Field(..., description="Creation timestamp")
    owned_by: str = Field(default="depth-anything-v2", description="Model owner")


class ModelsResponse(BaseModel):
    object: str = Field(default="list", description="Object type")
    data: List[ModelInfo] = Field(..., description="List of available models") 