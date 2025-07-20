import base64
import io
import numpy as np
import cv2
import requests
from PIL import Image
import torch
import os
import random

from depthany2.metric_main import DepthAny2
from depthany2.minimal_pts import depth2points


class DepthService:
    def __init__(self):
        self.models = {}  # {model_key: {gpu_id: model_instance}}
        self.request_counter = 0
        
        # Get all available GPUs
        self.gpu_count = torch.cuda.device_count()
        
        if self.gpu_count > 0:
            self.worker_id = self._get_worker_id()
            print(f"✅ Worker {self.worker_id} initialized with {self.gpu_count} GPUs available")
        else:
            self.worker_id = 0
            print("ℹ️  No GPUs available, using CPU")
    
    def _get_worker_id(self):
        """Get a unique worker ID (process ID) for this process."""
        return os.getpid()
    
    def _get_model_key(self, encoder, dataset, model_input_size):
        """Generate model key without max_depth (always use 1.0 for initialization)"""
        return f"{encoder}_{dataset}_{model_input_size}"
    
    def _get_model(self, encoder, dataset, model_input_size, gpu_id):
        """Get model instance for specific GPU"""
        key = self._get_model_key(encoder, dataset, model_input_size)
        
        # Initialize models dict for this key if not exists
        if key not in self.models:
            self.models[key] = {}
        
        # Initialize model on this GPU if not already done
        if gpu_id not in self.models[key]:
            device = torch.device(f'cuda:{gpu_id}')
            self.models[key][gpu_id] = DepthAny2(
                device=device,
                encoder=encoder,
                dataset=dataset,
                model_input_size=model_input_size,
                max_depth=1.0  # Always initialize with 1.0
            )
            print(f"✅ Worker {self.worker_id} loaded model {key} on cuda:{gpu_id}")
        
        return self.models[key][gpu_id]
    
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
        # Get image as numpy array (OpenCV format)
        img_array = self._get_image(image, image_url)
        
        # Randomly select GPU for this request
        if self.gpu_count > 0:
            gpu_id = random.randint(0, self.gpu_count - 1)
        else:
            gpu_id = None
        
        # Get model for this GPU
        model = self._get_model(encoder, dataset, model_input_size, gpu_id)
        
        # Get the device that the model is actually on
        device = model.DEVICE
        
        # Run prediction with numpy array (model expects OpenCV format)
        depth = model.predict(img_array, max_depth=max_depth)
        
        # Clean up to prevent memory leaks
        if self.gpu_count > 0:
            torch.cuda.empty_cache()
        
        return depth, device
    
    def predict_pointcloud(self, image=None, image_url=None, camera_intrinsics=None, 
                          encoder="vitl", dataset="hypersim", model_input_size=518, max_depth=1.0):
        self.request_counter += 1
        
        depth, device = self._predict_depth(image, image_url, encoder, dataset, model_input_size, max_depth)
        print(f"🔄 Worker {self.worker_id} processing pointcloud request on {device}")
        
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
        
        depth, device = self._predict_depth(image, image_url, encoder, dataset, model_input_size, max_depth)
        print(f"🔄 Worker {self.worker_id} processing metric_depth request on {device}")
        
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
        
        depth, device = self._predict_depth(image, image_url, encoder, dataset, model_input_size, max_depth=1.0)
        print(f"🔄 Worker {self.worker_id} processing relative_depth request on {device}")
        
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
        # Count models loaded by this worker across all GPUs
        total_models = sum(len(gpu_models) for gpu_models in self.models.values())
        model_keys = list(self.models.keys())
        
        # Determine device and gpu_id for reporting
        if self.gpu_count > 0:
            gpu_id = 0
            device = f"cuda:{gpu_id}"
        else:
            gpu_id = None
            device = "cpu"
        
        return {
            'status': 'healthy',
            'device': device,
            'worker_id': self.worker_id,
            'gpu_id': gpu_id,
            'gpu_count': self.gpu_count,
            'models_loaded': model_keys,
            'total_model_instances': total_models,
            'request_counter': self.request_counter
        } 