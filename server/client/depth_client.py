"""
OpenAI-style client for Depth Anything V2 API service.

This client provides a compatible interface with the standard OpenAI client
for depth prediction tasks.
"""
import os
import openai
import json
import yaml
import base64
import numpy as np
import cv2
from PIL import Image
import io
from typing import Dict, Any, Optional, Union, List


def depth_chat_completion(model: str, messages: List[Dict], base_url: str = None, 
                         api_key: str = None, **kwargs) -> openai.types.ChatCompletion:
    """
    Send a depth prediction request to the Depth Anything V2 API using OpenAI-compatible format.
    
    Args:
        model (str): Model name (e.g., "depth-anything-v2")
        messages (list): List of message dicts with image and camera intrinsics
        base_url (str): Base URL of the depth API endpoint
        api_key (str): API key (optional, for authentication)
        **kwargs: Additional parameters (response_format, temperature, etc.)
    
    Returns:
        openai.types.ChatCompletion: The response object
    """
    if base_url is None:
        base_url = "http://localhost:8000"
    
    # Initialize the OpenAI client
    client = openai.OpenAI(
        base_url=f"{base_url}/v1/depth",
        api_key=api_key or "dummy-key"  # API key not required for local service
    )
    
    # Create and send the chat completion request
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs
    )
    return response


def encode_image_to_base64(image_path: str) -> str:
    """Encode image file to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def encode_numpy_image_to_base64(image_array: np.ndarray) -> str:
    """Encode numpy image array to base64 string"""
    # Convert BGR to RGB if needed
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image_array
    
    # Convert to PIL Image and encode
    pil_image = Image.fromarray(image_rgb)
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def create_depth_request(image: Union[str, np.ndarray], 
                        camera_intrinsics: Dict[str, float],
                        encoder: str = "vitl",
                        dataset: str = "hypersim",
                        max_depth: float = 1.0,
                        model_input_size: int = 518,
                        response_format: str = "all") -> List[Dict]:
    """
    Create a depth prediction request in OpenAI-compatible format.
    
    Args:
        image: Image file path or numpy array
        camera_intrinsics: Dict with fx, fy, cx, cy
        encoder: Model encoder (vitl, vitb, vits)
        dataset: Dataset type (hypersim, vkitti)
        max_depth: Maximum depth value
        model_input_size: Model input size
        response_format: Response format (pointcloud, depth_relative, all)
    
    Returns:
        List of messages in OpenAI format
    """
    # Encode image
    if isinstance(image, str):
        image_base64 = encode_image_to_base64(image)
    else:
        image_base64 = encode_numpy_image_to_base64(image)
    
    # Create user content
    user_content = {
        "image": image_base64,
        "camera_intrinsics": camera_intrinsics,
        "options": {
            "encoder": encoder,
            "dataset": dataset,
            "max_depth": max_depth,
            "model_input_size": model_input_size
        }
    }
    
    # Create messages
    messages = [
        {
            "role": "user",
            "content": user_content
        }
    ]
    
    return messages


def decode_depth_response(response: openai.types.ChatCompletion) -> Dict[str, Any]:
    """
    Decode the depth prediction response.
    
    Args:
        response: OpenAI-compatible response from depth API
    
    Returns:
        Dict containing decoded depth maps and point clouds
    """
    content = response.choices[0].message.content
    
    result = {}
    
    # Decode depth map if present
    if 'depth_relative' in content:
        depth_data = content['depth_relative']['data']
        if depth_data.startswith('data:image/png;base64,'):
            depth_base64 = depth_data.split(',')[1]
        else:
            depth_base64 = depth_data
        
        depth_bytes = base64.b64decode(depth_base64)
        depth_image = Image.open(io.BytesIO(depth_bytes))
        result['depth_relative'] = np.array(depth_image)
        result['depth_relative_info'] = {
            'shape': content['depth_relative']['shape'],
            'min': content['depth_relative']['min'],
            'max': content['depth_relative']['max']
        }
    
    # Decode point cloud if present
    if 'pointcloud' in content:
        points_base64 = content['pointcloud']['data']
        points_bytes = base64.b64decode(points_base64)
        points_flat = np.frombuffer(points_bytes, dtype=np.float32)
        result['pointcloud'] = points_flat.reshape(content['pointcloud']['shape'])
    
    return result


def predict_depth_simple(image: Union[str, np.ndarray],
                        camera_intrinsics: Dict[str, float],
                        base_url: str = "http://localhost:8000",
                        encoder: str = "vitl",
                        dataset: str = "hypersim",
                        max_depth: float = 1.0,
                        model_input_size: int = 518,
                        format: str = "all") -> Dict[str, Any]:
    """
    Simple function to predict depth without OpenAI-compatible format.
    
    Args:
        image: Image file path or numpy array
        camera_intrinsics: Dict with fx, fy, cx, cy
        base_url: API base URL
        encoder: Model encoder
        dataset: Dataset type
        max_depth: Maximum depth value
        model_input_size: Model input size
        format: Response format
    
    Returns:
        Dict with prediction results
    """
    import requests
    
    # Encode image
    if isinstance(image, str):
        image_base64 = encode_image_to_base64(image)
    else:
        image_base64 = encode_numpy_image_to_base64(image)
    
    # Prepare request
    payload = {
        'image': image_base64,
        'fx': camera_intrinsics['fx'],
        'fy': camera_intrinsics['fy'],
        'cx': camera_intrinsics['cx'],
        'cy': camera_intrinsics['cy'],
        'encoder': encoder,
        'dataset': dataset,
        'max_depth': max_depth,
        'model_input_size': model_input_size,
        'format': format
    }
    
    # Send request
    response = requests.post(f"{base_url}/v1/depth/predict/simple", json=payload)
    response.raise_for_status()
    
    result = response.json()
    
    # Decode results if successful
    if result.get('success', False):
        decoded_result = {}
        
        if 'depth_relative' in result:
            depth_data = result['depth_relative']
            if depth_data.startswith('data:image/png;base64,'):
                depth_base64 = depth_data.split(',')[1]
            else:
                depth_base64 = depth_data
            
            depth_bytes = base64.b64decode(depth_base64)
            depth_image = Image.open(io.BytesIO(depth_bytes))
            decoded_result['depth_relative'] = np.array(depth_image)
        
        if 'pointcloud' in result:
            points_bytes = base64.b64decode(result['pointcloud'])
            points_flat = np.frombuffer(points_bytes, dtype=np.float32)
            decoded_result['pointcloud'] = points_flat
        
        result.update(decoded_result)
    
    return result


if __name__ == "__main__":
    # Example usage
    with open("../config/servers.yaml", "r") as f:
        servers = yaml.safe_load(f)
    
    SELECT = "depth-anything-v2"
    selected_server = servers[SELECT]
    
    # Example camera intrinsics (you should use your actual values)
    camera_intrinsics = {
        'fx': 1000.0,
        'fy': 1000.0,
        'cx': 640.0,
        'cy': 480.0
    }
    
    # Create request
    messages = create_depth_request(
        image="path/to/your/image.jpg",  # Replace with actual image path
        camera_intrinsics=camera_intrinsics,
        encoder="vitl",
        dataset="hypersim",
        response_format="all"
    )
    
    print(f"--- {SELECT} ---")
    try:
        response = depth_chat_completion(
            model="depth-anything-v2",
            messages=messages,
            base_url=selected_server["base_url"],
            api_key=selected_server.get("api_key"),
            response_format="all"
        )
        
        # Decode response
        result = decode_depth_response(response)
        print(f"Depth map shape: {result.get('depth_relative', 'Not available')}")
        print(f"Point cloud shape: {result.get('pointcloud', 'Not available')}")
        
    except Exception as e:
        print(f"Error: {e}") 