#!/usr/bin/env python3

import numpy as np
np.float = np.float64  # temp fix for following import https://github.com/eric-wieser/ros_numpy/issues/37
import torch
import os
from .metric_depth.depth_anything_v2.dpt import DepthAnythingV2


class DepthAny2:
    def __init__(self, device=None, model_input_size=518, max_depth=20, encoder='vitl', dataset='hypersim'):
        if device is not None:
            self.DEVICE = torch.device(device)
        else:
            self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.DEVICE == 'cuda':
            assert torch.cuda.is_available()
        self.model_input_size = model_input_size
        self.max_depth = max_depth  # 20 for indoor model, 80 for outdoor model
        self.encoder_name = encoder  # or 'vits', 'vitb', 'vitl'
        self.dataset_name = dataset  # 'hypersim' for indoor model, 'vkitti' for outdoor model
        self.setup_()

    def setup_(self):
        with torch.device(self.DEVICE):
            model_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
            }
            self.depth_model = DepthAnythingV2(**{**model_configs[self.encoder_name], 'max_depth': self.max_depth})
            ckpt_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), f'metric_depth/checkpoints/depth_anything_v2_metric_{self.dataset_name}_{self.encoder_name}.pth')
            self.depth_model.load_state_dict(torch.load(ckpt_path, map_location=self.DEVICE))
            self.depth_model.eval()

    @torch.inference_mode()
    def predict(self, cv2_img, max_depth=None):
        with torch.device(self.DEVICE):
            depth_arr = self.depth_model.infer_image(cv2_img, self.model_input_size, max_depth)
            # resized_pred = PILImage.fromarray(depth_arr).resize((cv2_img.shape[1], cv2_img.shape[0]), PILImage.NEAREST)   # not needed, already correct size
        return depth_arr

