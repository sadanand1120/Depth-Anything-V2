#!/usr/bin/env python3
"""Test script for Depth Anything V2 API"""

import sys
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt

# Add paths for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.client.depth_client import (
    encode_image, decode_pointcloud, decode_depth_map,
    predict_pointcloud, predict_metric_depth, predict_relative_depth, get_health
)
from depthany2.minimal_pts import load_intrinsics_from_yaml


def test_endpoints():
    """Test all three endpoints"""
    print("Testing Depth Anything V2 API endpoints...")
    
    # Load config
    with open("server/client/servers.yaml", "r") as f:
        servers = yaml.safe_load(f)
    
    base_url = servers["dany2"]["base_url"]
    api_key = servers["dany2"]["api_key"]
    
    # Load camera intrinsics (use example data if not available)
    try:
        intrinsics_path = '/home/dynamo/AMRL_Research/repos/robot_calib/spot/params/cam_intrinsics_3072.yaml'
        cam_intrinsics = load_intrinsics_from_yaml(intrinsics_path)
    except FileNotFoundError:
        print("⚠️  Using default camera intrinsics (example data not found)")
        # Default camera intrinsics for testing
        camera_intrinsics = {
            'fx': 1000.0, 'fy': 1000.0, 'cx': 640.0, 'cy': 480.0
        }
    else:
        camera_intrinsics = {
            'fx': cam_intrinsics['camera_matrix'][0, 0],
            'fy': cam_intrinsics['camera_matrix'][1, 1],
            'cx': cam_intrinsics['camera_matrix'][0, 2],
            'cy': cam_intrinsics['camera_matrix'][1, 2]
        }
    
    # Image path (use example data if not available)
    try:
        image_path = '/home/dynamo/AMRL_Research/repos/robot_calib/notrack_bags/paths/poc_cut/images/1742591008774347085.png'
        image_base64 = encode_image(image_path)
    except FileNotFoundError:
        print("⚠️  Example image not found. Please provide a valid image path.")
        print("   You can use any image file and update the image_path variable.")
        return False
    
    try:
        # Test health
        health = get_health(base_url, api_key)
        print(f"✅ Health: {health}")
        
        # Test pointcloud
        pc_result = predict_pointcloud(
            image=image_base64,
            camera_intrinsics=camera_intrinsics,
            base_url=base_url,
            max_depth=1.0,
            api_key=api_key
        )
        print(f"✅ Pointcloud: shape {pc_result['pointcloud_shape']}")
        
        # Test metric depth
        metric_result = predict_metric_depth(
            image=image_base64,
            base_url=base_url,
            max_depth=1.0,
            api_key=api_key
        )
        print(f"✅ Metric depth: shape {metric_result['shape']}, range [{metric_result['min']:.3f}, {metric_result['max']:.3f}]")
        
        # Test relative depth
        rel_result = predict_relative_depth(
            image=image_base64,
            base_url=base_url,
            api_key=api_key
        )
        print(f"✅ Relative depth: shape {rel_result['shape']}, range [{rel_result['min']:.3f}, {rel_result['max']:.3f}]")
        
        # Decode and visualize
        pointcloud = decode_pointcloud(pc_result['pointcloud'], pc_result['pointcloud_shape'])
        depth_map = decode_depth_map(metric_result['depth_map'], metric_result['shape'])
        
        print(f"✅ Decoded pointcloud: {pointcloud.shape}")
        print(f"✅ Decoded depth map: {depth_map.shape}")
        
        # Visualize depth map
        plt.figure(figsize=(10, 8))
        plt.imshow(depth_map, cmap='viridis')
        plt.title("Predicted Depth Map")
        plt.colorbar()
        plt.show()
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_authentication():
    """Test API key authentication"""
    print("\n🔐 Testing API Key Authentication...")
    
    # Load config
    with open("server/client/servers.yaml", "r") as f:
        servers = yaml.safe_load(f)
    
    base_url = servers["dany2"]["base_url"]
    
    # Test without API key (should work if auth is disabled)
    try:
        health = get_health(base_url)
        print("✅ Health check without API key: SUCCESS")
    except Exception as e:
        print(f"❌ Health check without API key: {e}")
    
    # Test with invalid API key
    try:
        health = get_health(base_url, api_key="invalid-key")
        print("❌ Health check with invalid API key: Should have failed")
    except Exception as e:
        if "401" in str(e) or "Unauthorized" in str(e):
            print("✅ Health check with invalid API key: Correctly rejected")
        else:
            print(f"❌ Health check with invalid API key: Unexpected error - {e}")
    
    # Test with valid API key (if configured)
    valid_api_key = servers["dany2"]["api_key"]
    if valid_api_key and valid_api_key != "smdepth":
        try:
            health = get_health(base_url, api_key=valid_api_key)
            print("✅ Health check with valid API key: SUCCESS")
        except Exception as e:
            print(f"❌ Health check with valid API key: {e}")
    else:
        print("ℹ️  Skipping valid API key test (not configured)")


if __name__ == "__main__":
    test_endpoints()
    test_authentication() 