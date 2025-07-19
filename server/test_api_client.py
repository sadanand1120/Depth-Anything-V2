#!/usr/bin/env python3
"""
Test script for the Depth Anything V2 API client.

This script demonstrates how to use the API client with the same functionality
as the original minimal_pts.py script.
"""

import sys
import os
import yaml
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Add the current directory and parent directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client.depth_client import (
    depth_chat_completion, 
    create_depth_request, 
    decode_depth_response,
    predict_depth_simple
)
from depthany2.minimal_pts import load_intrinsics_from_yaml


def test_openai_style_client():
    """Test the OpenAI-style client"""
    print("Testing OpenAI-style client...")
    
    # Load server configuration
    with open("config/servers.yaml", "r") as f:
        servers = yaml.safe_load(f)
    
    SELECT = "depth-anything-v2-local"
    selected_server = servers[SELECT]
    
    # Load camera intrinsics (using the same file as in minimal_pts.py)
    intrinsics_path = '/home/dynamo/AMRL_Research/repos/robot_calib/spot/params/cam_intrinsics_3072.yaml'
    cam_intrinsics = load_intrinsics_from_yaml(intrinsics_path)
    
    # Convert to the format expected by the API
    camera_intrinsics = {
        'fx': cam_intrinsics['camera_matrix'][0, 0],
        'fy': cam_intrinsics['camera_matrix'][1, 1],
        'cx': cam_intrinsics['camera_matrix'][0, 2],
        'cy': cam_intrinsics['camera_matrix'][1, 2]
    }
    
    # Image path (using the same as in minimal_pts.py)
    image_path = '/home/dynamo/AMRL_Research/repos/robot_calib/notrack_bags/paths/poc_cut/images/1742591008774347085.png'
    
    # Create request
    messages = create_depth_request(
        image=image_path,
        camera_intrinsics=camera_intrinsics,
        encoder="vitl",
        dataset="hypersim",
        max_depth=1.0,
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
        
        print(f"Success! Received response:")
        print(f"  - Depth map shape: {result.get('depth_relative', 'Not available')}")
        print(f"  - Point cloud shape: {result.get('pointcloud', 'Not available')}")
        
        # Visualize depth map
        if 'depth_relative' in result:
            plt.figure(figsize=(10, 8))
            plt.imshow(result['depth_relative'], cmap='viridis')
            plt.title("Predicted Relative Depth (API)")
            plt.colorbar()
            plt.show()
        
        return result
        
    except Exception as e:
        print(f"Error: {e}")
        return None


def test_simple_client():
    """Test the simple client"""
    print("\nTesting simple client...")
    
    # Load camera intrinsics
    intrinsics_path = '/home/dynamo/AMRL_Research/repos/robot_calib/spot/params/cam_intrinsics_3072.yaml'
    cam_intrinsics = load_intrinsics_from_yaml(intrinsics_path)
    
    camera_intrinsics = {
        'fx': cam_intrinsics['camera_matrix'][0, 0],
        'fy': cam_intrinsics['camera_matrix'][1, 1],
        'cx': cam_intrinsics['camera_matrix'][0, 2],
        'cy': cam_intrinsics['camera_matrix'][1, 2]
    }
    
    image_path = '/home/dynamo/AMRL_Research/repos/robot_calib/notrack_bags/paths/poc_cut/images/1742591008774347085.png'
    
    try:
        result = predict_depth_simple(
            image=image_path,
            camera_intrinsics=camera_intrinsics,
            base_url="http://localhost:8000",
            encoder="vitl",
            dataset="hypersim",
            max_depth=1.0,
            format="all"
        )
        
        if result.get('success', False):
            print(f"Success! Received response:")
            print(f"  - Image shape: {result.get('image_shape', 'Not available')}")
            print(f"  - Point cloud shape: {result.get('pointcloud_shape', 'Not available')}")
            
            # Visualize depth map
            if 'depth_relative' in result:
                plt.figure(figsize=(10, 8))
                plt.imshow(result['depth_relative'], cmap='viridis')
                plt.title("Predicted Relative Depth (Simple API)")
                plt.colorbar()
                plt.show()
            
            return result
        else:
            print(f"API returned error: {result.get('error', 'Unknown error')}")
            return None
            
    except Exception as e:
        print(f"Error: {e}")
        return None


def compare_with_original():
    """Compare API results with original minimal_pts.py"""
    print("\nComparing with original minimal_pts.py...")
    
    # Import original function
    from depthany2.minimal_pts import predict_depth_and_points, load_intrinsics_from_yaml
    
    # Load camera intrinsics
    intrinsics_path = '/home/dynamo/AMRL_Research/repos/robot_calib/spot/params/cam_intrinsics_3072.yaml'
    cam_intrinsics = load_intrinsics_from_yaml(intrinsics_path)
    
    image_path = '/home/dynamo/AMRL_Research/repos/robot_calib/notrack_bags/paths/poc_cut/images/1742591008774347085.png'
    
    try:
        # Original prediction
        points_orig, depth_orig, img_orig = predict_depth_and_points(
            image_path, cam_intrinsics, encoder="vitl", dataset="hypersim",
            model_input_size=518, max_depth=1.0, return_image=True
        )
        
        print(f"Original results:")
        print(f"  - Depth map shape: {depth_orig.shape}")
        print(f"  - Point cloud shape: {points_orig.shape}")
        
        # API prediction (if server is running)
        api_result = test_simple_client()
        
        if api_result and 'depth_relative' in api_result:
            print(f"\nAPI results:")
            print(f"  - Depth map shape: {api_result['depth_relative'].shape}")
            print(f"  - Point cloud shape: {api_result['pointcloud'].shape}")
            
            # Compare shapes
            depth_match = depth_orig.shape == api_result['depth_relative'].shape
            points_match = points_orig.shape == api_result['pointcloud'].shape
            
            print(f"\nShape comparison:")
            print(f"  - Depth maps match: {depth_match}")
            print(f"  - Point clouds match: {points_match}")
            
            if depth_match and points_match:
                print("✅ API results match original implementation!")
            else:
                print("❌ API results differ from original implementation")
        
    except Exception as e:
        print(f"Error in comparison: {e}")


def main():
    """Main test function"""
    print("Depth Anything V2 API Client Test")
    print("=" * 50)
    
    # Test 1: OpenAI-style client
    test_openai_style_client()
    
    # Test 2: Simple client
    test_simple_client()
    
    # Test 3: Comparison with original
    compare_with_original()
    
    print("\nTest completed!")


if __name__ == "__main__":
    main() 