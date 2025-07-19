"""
OpenAI-style client for Depth Anything V2 API service.

This client provides a compatible interface with the standard OpenAI client
for depth prediction tasks.
"""
import base64
import numpy as np
import cv2
import requests
from PIL import Image
import io
from typing import Dict, Optional, Union


def encode_image(image_path: str) -> str:
    """Encode image file to base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def decode_pointcloud(pointcloud_base64: str, shape: list) -> np.ndarray:
    """Decode base64 pointcloud to numpy array"""
    points_bytes = base64.b64decode(pointcloud_base64)
    points_flat = np.frombuffer(points_bytes, dtype=np.float32)
    return points_flat.reshape(shape)


def decode_depth_map(depth_base64: str) -> np.ndarray:
    """Decode base64 depth map to numpy array"""
    if depth_base64.startswith('data:image/png;base64,'):
        depth_base64 = depth_base64.split(',')[1]
    depth_bytes = base64.b64decode(depth_base64)
    depth_image = Image.open(io.BytesIO(depth_bytes))
    return np.array(depth_image)


def predict_pointcloud(image: Optional[str] = None, image_url: Optional[str] = None,
                      camera_intrinsics: Optional[Dict] = None, base_url: str = "http://localhost:8000",
                      encoder: str = "vitl", dataset: str = "hypersim", model_input_size: int = 518,
                      max_depth: float = 1.0) -> Dict:
    """Get pointcloud from depth prediction"""
    payload = {
        'camera_intrinsics': camera_intrinsics,
        'encoder': encoder,
        'dataset': dataset,
        'model_input_size': model_input_size,
        'max_depth': max_depth
    }
    
    if image:
        payload['image'] = image
    elif image_url:
        payload['image_url'] = image_url
    else:
        raise ValueError("Either image or image_url must be provided")
    
    response = requests.post(f"{base_url}/pc", json=payload)
    response.raise_for_status()
    return response.json()


def predict_metric_depth(image: Optional[str] = None, image_url: Optional[str] = None,
                        camera_intrinsics: Optional[Dict] = None, base_url: str = "http://localhost:8000",
                        encoder: str = "vitl", dataset: str = "hypersim", model_input_size: int = 518,
                        max_depth: float = 1.0) -> Dict:
    """Get metric depth map from depth prediction"""
    payload = {
        'camera_intrinsics': camera_intrinsics,
        'encoder': encoder,
        'dataset': dataset,
        'model_input_size': model_input_size,
        'max_depth': max_depth
    }
    
    if image:
        payload['image'] = image
    elif image_url:
        payload['image_url'] = image_url
    else:
        raise ValueError("Either image or image_url must be provided")
    
    response = requests.post(f"{base_url}/metric_depth", json=payload)
    response.raise_for_status()
    return response.json()


def predict_relative_depth(image: Optional[str] = None, image_url: Optional[str] = None,
                          camera_intrinsics: Optional[Dict] = None, base_url: str = "http://localhost:8000",
                          encoder: str = "vitl", dataset: str = "hypersim", model_input_size: int = 518) -> Dict:
    """Get relative depth map from depth prediction"""
    payload = {
        'camera_intrinsics': camera_intrinsics,
        'encoder': encoder,
        'dataset': dataset,
        'model_input_size': model_input_size
    }
    
    if image:
        payload['image'] = image
    elif image_url:
        payload['image_url'] = image_url
    else:
        raise ValueError("Either image or image_url must be provided")
    
    response = requests.post(f"{base_url}/rel_depth", json=payload)
    response.raise_for_status()
    return response.json()


def get_health(base_url: str = "http://localhost:8000") -> Dict:
    """Get server health status"""
    response = requests.get(f"{base_url}/health")
    response.raise_for_status()
    return response.json() 