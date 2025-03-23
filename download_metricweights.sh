#!/bin/bash

cd metric_depth/
mkdir -p checkpoints/
cd checkpoints/
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth -O depth_anything_v2_vitl.pth
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth -O depth_anything_v2_vits.pth
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Large/resolve/main/depth_anything_v2_metric_vkitti_vitl.pth -O depth_anything_v2_metric_vkitti_vitl.pth
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Small/resolve/main/depth_anything_v2_metric_vkitti_vits.pth -O depth_anything_v2_metric_vkitti_vits.pth
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large/resolve/main/depth_anything_v2_metric_hypersim_vitl.pth -O depth_anything_v2_metric_hypersim_vitl.pth
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/resolve/main/depth_anything_v2_metric_hypersim_vits.pth -O depth_anything_v2_metric_hypersim_vits.pth
cd ../..