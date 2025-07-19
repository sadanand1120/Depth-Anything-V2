import base64
import io
import numpy as np
import cv2
import requests
from PIL import Image
import torch
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from depthany2.metric_main import DepthAny2
from depthany2.minimal_pts import depth2points


class DepthService:
    def __init__(self):
        self.models = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _get_model(self, encoder, dataset, model_input_size, max_depth):
        key = f"{encoder}_{dataset}_{model_input_size}_{max_depth}"
        if key not in self.models:
            self.models[key] = DepthAny2(
                device=self.device,
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
    
    def predict_pointcloud(self, image=None, image_url=None, camera_intrinsics=None, 
                          encoder="vitl", dataset="hypersim", model_input_size=518, max_depth=1.0):
        img_array = self._get_image(image, image_url)
        model = self._get_model(encoder, dataset, model_input_size, max_depth)
        depth = model.predict(img_array, max_depth=max_depth)
        points = depth2points(depth, self._camera_intrinsics_to_dict(camera_intrinsics))
        
        # Encode as base64
        points_flat = points.astype(np.float32)
        points_base64 = base64.b64encode(points_flat.tobytes()).decode('utf-8')
        
        return {
            'pointcloud': points_base64,
            'shape': points.shape
        }
    
    def predict_metric_depth(self, image=None, image_url=None, camera_intrinsics=None,
                            encoder="vitl", dataset="hypersim", model_input_size=518, max_depth=1.0):
        img_array = self._get_image(image, image_url)
        model = self._get_model(encoder, dataset, model_input_size, max_depth)
        depth = model.predict(img_array, max_depth=max_depth)
        
        # Encode as base64
        depth_normalized = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
        _, depth_buffer = cv2.imencode('.png', depth_normalized)
        depth_base64 = base64.b64encode(depth_buffer).decode('utf-8')
        
        return {
            'depth_map': f"data:image/png;base64,{depth_base64}",
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
            'models_loaded': list(self.models.keys())
        }


# Global service instance
depth_service = DepthService() 