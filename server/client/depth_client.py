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


def decode_depth_map(depth_base64: str, shape: Optional[list] = None) -> np.ndarray:
    """Decode base64 depth map to numpy array"""
    # Handle raw float32 depth data (new format)
    depth_bytes = base64.b64decode(depth_base64)
    depth_flat = np.frombuffer(depth_bytes, dtype=np.float32)
    
    # Reshape if shape is provided
    if shape:
        return depth_flat.reshape(shape)
    return depth_flat


def _get_headers(api_key: Optional[str] = None) -> Dict[str, str]:
    """Get headers for API requests"""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def predict_pointcloud(image: Optional[str] = None, image_url: Optional[str] = None,
                      camera_intrinsics: Optional[Dict] = None, base_url: str = "http://localhost:8000",
                      encoder: str = "vitl", dataset: str = "hypersim", model_input_size: int = 518,
                      max_depth: float = 1.0, api_key: Optional[str] = None) -> Dict:
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
    
    headers = _get_headers(api_key)
    response = requests.post(f"{base_url}/pc", json=payload, headers=headers)
    response.raise_for_status()
    return response.json()


def predict_metric_depth(image: Optional[str] = None, image_url: Optional[str] = None,
                        base_url: str = "http://localhost:8000",
                        encoder: str = "vitl", dataset: str = "hypersim", model_input_size: int = 518,
                        max_depth: float = 1.0, api_key: Optional[str] = None) -> Dict:
    """Get metric depth map from depth prediction"""
    payload = {
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
    
    headers = _get_headers(api_key)
    response = requests.post(f"{base_url}/metric_depth", json=payload, headers=headers)
    response.raise_for_status()
    return response.json()


def predict_relative_depth(image: Optional[str] = None, image_url: Optional[str] = None,
                          base_url: str = "http://localhost:8000",
                          encoder: str = "vitl", dataset: str = "hypersim", model_input_size: int = 518,
                          api_key: Optional[str] = None) -> Dict:
    """Get relative depth map from depth prediction"""
    payload = {
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
    
    headers = _get_headers(api_key)
    response = requests.post(f"{base_url}/rel_depth", json=payload, headers=headers)
    response.raise_for_status()
    return response.json()


def get_health(base_url: str = "http://localhost:8000", api_key: Optional[str] = None) -> Dict:
    """Get server health status"""
    headers = _get_headers(api_key)
    response = requests.get(f"{base_url}/health", headers=headers)
    response.raise_for_status()
    return response.json() 