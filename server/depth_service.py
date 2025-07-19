import base64
import io
import time
import uuid
import numpy as np
import cv2
from PIL import Image
import torch
from typing import Dict, Any, Optional, Tuple
import sys
import os

# Add the parent directory to path to import depthany2
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from depthany2.metric_main import DepthAny2
from depthany2.minimal_pts import depth2points, load_intrinsics_from_yaml


class DepthPredictionService:
    def __init__(self):
        self.models = {}  # Cache for loaded models
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _get_model_key(self, encoder: str, dataset: str, model_input_size: int, max_depth: float) -> str:
        """Generate a unique key for model caching"""
        return f"{encoder}_{dataset}_{model_input_size}_{max_depth}"
    
    def _load_model(self, encoder: str, dataset: str, model_input_size: int, max_depth: float) -> DepthAny2:
        """Load and cache a depth prediction model"""
        model_key = self._get_model_key(encoder, dataset, model_input_size, max_depth)
        
        if model_key not in self.models:
            print(f"Loading model: {model_key}")
            self.models[model_key] = DepthAny2(
                device=self.device,
                encoder=encoder,
                dataset=dataset,
                model_input_size=model_input_size,
                max_depth=max_depth
            )
            print(f"Model loaded successfully: {model_key}")
        
        return self.models[model_key]
    
    def _decode_base64_image(self, image_base64: str) -> np.ndarray:
        """Decode base64 image to numpy array"""
        try:
            # Remove data URL prefix if present
            if image_base64.startswith('data:image'):
                image_base64 = image_base64.split(',')[1]
            
            # Decode base64
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to numpy array (BGR for OpenCV)
            image_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            return image_array
        except Exception as e:
            raise ValueError(f"Failed to decode base64 image: {str(e)}")
    
    def _camera_intrinsics_to_dict(self, intrinsics: Dict[str, float]) -> Dict[str, np.ndarray]:
        """Convert camera intrinsics to the format expected by depth2points"""
        fx, fy, cx, cy = intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy']
        camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        return {'camera_matrix': camera_matrix}
    
    def predict(self, 
                image_base64: str,
                camera_intrinsics: Dict[str, float],
                encoder: str = "vitl",
                dataset: str = "hypersim",
                model_input_size: int = 518,
                max_depth: float = 1.0) -> Dict[str, Any]:
        """
        Predict depth and generate point cloud from base64 encoded image
        
        Returns:
            Dict containing:
            - depth_relative: relative depth map
            - depth_metric: metric depth map (if available)
            - pointcloud: 3D points
            - original_image: original image array
        """
        # Decode image
        image_array = self._decode_base64_image(image_base64)
        
        # Load model
        model = self._load_model(encoder, dataset, model_input_size, max_depth)
        
        # Predict depth
        depth_relative = model.predict(image_array, max_depth=max_depth)
        
        # Convert camera intrinsics
        cam_intrinsics_dict = self._camera_intrinsics_to_dict(camera_intrinsics)
        
        # Generate point cloud
        points = depth2points(depth_relative, cam_intrinsics_dict)
        
        return {
            'depth_relative': depth_relative,
            'pointcloud': points,
            'original_image': image_array,
            'image_shape': image_array.shape,
            'pointcloud_shape': points.shape
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get service health status"""
        return {
            'status': 'healthy',
            'model_loaded': len(self.models) > 0,
            'device': str(self.device),
            'cuda_available': torch.cuda.is_available(),
            'loaded_models': list(self.models.keys())
        }
    
    def get_available_models(self) -> Dict[str, Any]:
        """Get available model configurations"""
        return {
            'models': [
                {
                    'id': 'depth-anything-v2-vitl-hypersim',
                    'object': 'model',
                    'created': int(time.time()),
                    'owned_by': 'depth-anything-v2',
                    'config': {
                        'encoder': 'vitl',
                        'dataset': 'hypersim',
                        'model_input_size': 518,
                        'max_depth': 1.0
                    }
                },
                {
                    'id': 'depth-anything-v2-vitb-hypersim',
                    'object': 'model',
                    'created': int(time.time()),
                    'owned_by': 'depth-anything-v2',
                    'config': {
                        'encoder': 'vitb',
                        'dataset': 'hypersim',
                        'model_input_size': 518,
                        'max_depth': 1.0
                    }
                },
                {
                    'id': 'depth-anything-v2-vits-hypersim',
                    'object': 'model',
                    'created': int(time.time()),
                    'owned_by': 'depth-anything-v2',
                    'config': {
                        'encoder': 'vits',
                        'dataset': 'hypersim',
                        'model_input_size': 518,
                        'max_depth': 1.0
                    }
                }
            ]
        }


# Global service instance
depth_service = DepthPredictionService() 