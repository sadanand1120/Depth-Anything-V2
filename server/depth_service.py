import base64
import io
import numpy as np
import cv2
import requests
from PIL import Image
import torch
import os

from depthany2.metric_main import DepthAny2
from depthany2.minimal_pts import depth2points


class DepthService:
    def __init__(self):
        self.models = {}  # {model_key: model_instance}
        self.request_counter = 0
        
        # Get all available GPUs
        self.gpu_count = torch.cuda.device_count()
        
        if self.gpu_count > 0:
            # Determine which GPU this worker should use
            self.worker_id = self._get_worker_id()
            self.gpu_id = self.worker_id % self.gpu_count
            self.device = torch.device(f'cuda:{self.gpu_id}')
            print(f"✅ Worker {self.worker_id} using GPU {self.gpu_id} (cuda:{self.gpu_id})")
        else:
            self.worker_id = 0
            self.gpu_id = None
            self.device = torch.device('cpu')
            print("ℹ️  No GPUs available, using CPU")
    
    def _get_worker_id(self):
        """Get worker ID from environment or process info"""
        # Try to get from environment variable set by uvicorn
        worker_id = os.environ.get('UVICORN_WORKER_ID')
        if worker_id:
            return int(worker_id)
        
        # Fallback: use process ID modulo number of GPUs
        return os.getpid() % max(1, self.gpu_count)
    
    def _get_model_key(self, encoder, dataset, model_input_size):
        """Generate model key without max_depth (always use 1.0 for initialization)"""
        return f"{encoder}_{dataset}_{model_input_size}"
    
    def _get_model(self, encoder, dataset, model_input_size):
        """Get model instance for this worker's assigned GPU"""
        key = self._get_model_key(encoder, dataset, model_input_size)
        
        # Initialize model if not already done
        if key not in self.models:
            self.models[key] = DepthAny2(
                device=self.device,
                encoder=encoder,
                dataset=dataset,
                model_input_size=model_input_size,
                max_depth=1.0  # Always initialize with 1.0
            )
            print(f"✅ Worker {self.worker_id} loaded model {key} on {self.device}")
        
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
        model = self._get_model(encoder, dataset, model_input_size)
        # Override max_depth at runtime (model was initialized with 1.0)
        depth = model.predict(img_array, max_depth=max_depth)
        return depth
    
    def predict_pointcloud(self, image=None, image_url=None, camera_intrinsics=None, 
                          encoder="vitl", dataset="hypersim", model_input_size=518, max_depth=1.0):
        self.request_counter += 1
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
    
    def predict_metric_depth(self, image=None, image_url=None,
                            encoder="vitl", dataset="hypersim", model_input_size=518, max_depth=1.0):
        self.request_counter += 1
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
    
    def predict_relative_depth(self, image=None, image_url=None,
                              encoder="vitl", dataset="hypersim", model_input_size=518):
        self.request_counter += 1
        depth = self._predict_depth(image, image_url, encoder, dataset, model_input_size, max_depth=1.0)
        
        # Return raw relative depth values (max_depth=1.0)
        depth_flat = depth.astype(np.float32)
        depth_base64 = base64.b64encode(depth_flat.tobytes()).decode('utf-8')
        
        return {
            'depth_map': depth_base64,
            'shape': depth.shape,
            'min': float(depth.min()),
            'max': float(depth.max())
        }
    
    def get_health(self):
        # Count models loaded by this worker
        model_keys = list(self.models.keys())
        
        return {
            'status': 'healthy',
            'device': str(self.device),
            'worker_id': self.worker_id,
            'gpu_id': self.gpu_id if self.gpu_count > 0 else None,
            'gpu_count': self.gpu_count,
            'models_loaded': model_keys,
            'total_model_instances': len(self.models),
            'request_counter': self.request_counter
        } 