import base64
import io
import numpy as np
import cv2
import requests
from PIL import Image
import torch
import sys
import os

from depthany2.metric_main import DepthAny2
from depthany2.minimal_pts import depth2points


class DepthService:
    def __init__(self):
        self.models = {}
        # Get all available GPUs from CUDA_VISIBLE_DEVICES
        self.gpu_count = torch.cuda.device_count()
        self.current_gpu = 0
        
        if self.gpu_count > 0:
            self.device = torch.device('cuda')
            print(f"✅ Using {self.gpu_count} GPU(s): {[f'cuda:{i}' for i in range(self.gpu_count)]}")
        else:
            self.device = torch.device('cpu')
            print("ℹ️  No GPUs available, using CPU")
    
    def _get_next_gpu(self):
        """Get next GPU in round-robin fashion"""
        if self.gpu_count == 0:
            return self.device
        gpu_id = self.current_gpu % self.gpu_count
        self.current_gpu += 1
        return torch.device(f'cuda:{gpu_id}')
    
    def _get_model(self, encoder, dataset, model_input_size, max_depth):
        key = f"{encoder}_{dataset}_{model_input_size}_{max_depth}"
        if key not in self.models:
            gpu_device = self._get_next_gpu()
            self.models[key] = DepthAny2(
                device=gpu_device,
                encoder=encoder,
                dataset=dataset,
                model_input_size=model_input_size,
                max_depth=max_depth
            )
        return self.models[key]
    
    def _get_image(self, image=None, image_url=None):
        if image:
            # Remove data URL prefix if present
            if image.startswith('data:image'):
                image = image.split(',')[1]
            image_data = base64.b64decode(image)
            img = Image.open(io.BytesIO(image_data))
            return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        elif image_url:
            response = requests.get(image_url)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content))
            return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        raise ValueError("Either image or image_url must be provided")
    
    def _camera_intrinsics_to_dict(self, intrinsics):
        return {
            'camera_matrix': np.array([
                [intrinsics.fx, 0, intrinsics.cx],
                [0, intrinsics.fy, intrinsics.cy],
                [0, 0, 1]
            ])
        }
    
    def _predict_depth(self, image=None, image_url=None, encoder="vitl", dataset="hypersim", 
                      model_input_size=518, max_depth=1.0):
        """Shared method for depth prediction - eliminates redundant code"""
        img_array = self._get_image(image, image_url)
        model = self._get_model(encoder, dataset, model_input_size, max_depth)
        depth = model.predict(img_array, max_depth=max_depth)
        return depth
    
    def predict_pointcloud(self, image=None, image_url=None, camera_intrinsics=None, 
                          encoder="vitl", dataset="hypersim", model_input_size=518, max_depth=1.0):
        depth = self._predict_depth(image, image_url, encoder, dataset, model_input_size, max_depth)
        points = depth2points(depth, self._camera_intrinsics_to_dict(camera_intrinsics))
        
        # Encode pointcloud as base64
        points_flat = points.astype(np.float32)
        points_base64 = base64.b64encode(points_flat.tobytes()).decode('utf-8')
        
        # Encode depth as base64 for consistency with other endpoints
        depth_flat = depth.astype(np.float32)
        depth_base64 = base64.b64encode(depth_flat.tobytes()).decode('utf-8')
        
        return {
            'pointcloud': points_base64,
            'pointcloud_shape': points.shape,
            'depth_map': depth_base64,
            'depth_shape': depth.shape,
            'min': float(depth.min()),
            'max': float(depth.max())
        }
    
    def predict_metric_depth(self, image=None, image_url=None, camera_intrinsics=None,
                            encoder="vitl", dataset="hypersim", model_input_size=518, max_depth=1.0):
        depth = self._predict_depth(image, image_url, encoder, dataset, model_input_size, max_depth)
        
        # Return raw metric depth values (no normalization)
        depth_flat = depth.astype(np.float32)
        depth_base64 = base64.b64encode(depth_flat.tobytes()).decode('utf-8')
        
        return {
            'depth_map': depth_base64,
            'shape': depth.shape,
            'min': float(depth.min()),
            'max': float(depth.max())
        }
    
    def predict_relative_depth(self, image=None, image_url=None, camera_intrinsics=None,
                              encoder="vitl", dataset="hypersim", model_input_size=518):
        return self.predict_metric_depth(image, image_url, camera_intrinsics, 
                                       encoder, dataset, model_input_size, max_depth=1.0)
    
    def get_health(self):
        return {
            'status': 'healthy',
            'device': str(self.device),
            'gpu_count': self.gpu_count,
            'models_loaded': list(self.models.keys())
        } 