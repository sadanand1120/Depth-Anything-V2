#!/usr/bin/env python3
"""Test script for Depth Anything V2 API"""

import sys
import os
import yaml
import numpy as np

import matplotlib.pyplot as plt

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
        intrinsics_path = 'server/example/cam_intrinsics_3072.yaml'
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
        image_path = 'server/example/ahg_courtyard.png'
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
        
        # Set matplotlib backend to non-interactive to avoid X11 errors
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend

        # Visualize depth map
        plt.figure(figsize=(10, 8))
        plt.imshow(depth_map, cmap='viridis')
        plt.title("Predicted Depth Map")
        plt.colorbar()
        plt.savefig("depth_map.png")
        print("✅ Depth map saved to depth_map.png")
        
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
    valid_api_key = servers["dany2"]["api_key"]
    
    # Test health endpoint (no auth required)
    try:
        health = get_health(base_url)
        print("✅ Health check without API key: SUCCESS")
    except Exception as e:
        print(f"❌ Health check without API key: {e}")
    
    # Test with valid API key
    try:
        health = get_health(base_url, api_key=valid_api_key)
        print("✅ Health check with valid API key: SUCCESS")
    except Exception as e:
        print(f"❌ Health check with valid API key: {e}")
    
    # Test protected endpoint with invalid API key
    try:
        import requests
        
        # Use the same image from main test
        image_path = 'server/example/ahg_courtyard.png'
        image_base64 = encode_image(image_path)
        test_payload = {'image': image_base64, 'max_depth': 1.0}
        
        # Test invalid API key
        headers = {"Content-Type": "application/json", "Authorization": "Bearer invalid-key"}
        response = requests.post(f"{base_url}/metric_depth", json=test_payload, headers=headers)
        if response.status_code == 401:
            print("✅ Protected endpoint correctly rejected invalid API key")
        else:
            print(f"❌ Protected endpoint should have rejected invalid API key, got {response.status_code}")
        
        # Test no API key
        headers_no_auth = {"Content-Type": "application/json"}
        response_no_auth = requests.post(f"{base_url}/metric_depth", json=test_payload, headers=headers_no_auth)
        if response_no_auth.status_code == 401:
            print("✅ Protected endpoint correctly rejected request without API key")
        else:
            print(f"❌ Protected endpoint should have rejected request without API key, got {response_no_auth.status_code}")
            
    except Exception as e:
        print(f"❌ Authentication test error: {e}")


def test_image_url():
    """Test endpoint with image URL instead of base64"""
    print("\n🌐 Testing Image URL Endpoint...")
    
    # Load config
    with open("server/client/servers.yaml", "r") as f:
        servers = yaml.safe_load(f)
    
    base_url = servers["dany2"]["base_url"]
    api_key = servers["dany2"]["api_key"]
    
    # Use a random image from the internet
    image_url = "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=640&h=480&fit=crop"
    
    try:
        # Test metric depth with image URL
        metric_result = predict_metric_depth(
            image_url=image_url,
            base_url=base_url,
            max_depth=1.0,
            api_key=api_key
        )
        print(f"✅ Image URL metric depth: shape {metric_result['shape']}, range [{metric_result['min']:.3f}, {metric_result['max']:.3f}]")
        
        # Test relative depth with image URL
        rel_result = predict_relative_depth(
            image_url=image_url,
            base_url=base_url,
            api_key=api_key
        )
        print(f"✅ Image URL relative depth: shape {rel_result['shape']}, range [{rel_result['min']:.3f}, {rel_result['max']:.3f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ Image URL test error: {e}")
        return False


def test_concurrent_queries():
    """Test concurrent requests to verify GPU utilization"""
    print("\n⚡ Testing Concurrent Requests...")
    
    # Load config
    with open("server/client/servers.yaml", "r") as f:
        servers = yaml.safe_load(f)
    
    base_url = servers["dany2"]["base_url"]
    api_key = servers["dany2"]["api_key"]
    
    # Get test image
    try:
        image_path = 'server/example/ahg_courtyard.png'
        image_base64 = encode_image(image_path)
    except FileNotFoundError:
        print("⚠️  Example image not found, skipping concurrent test")
        return False
    
    import asyncio
    import aiohttp
    import time
    import json
    
    async def make_async_request(session, request_id, image_base64, base_url, api_key):
        """Make a single async request"""
        try:
            request_start = time.time()
            
            payload = {
                'image': image_base64,
                'max_depth': 1.0
            }
            
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            
            async with session.post(f"{base_url}/metric_depth", json=payload, headers=headers) as response:
                if response.status == 200:
                    result = await response.json()
                    request_time = time.time() - request_start
                    return {
                        'request_id': request_id,
                        'success': True,
                        'time': request_time,
                        'shape': result['shape']
                    }
                else:
                    error_text = await response.text()
                    return {
                        'request_id': request_id,
                        'success': False,
                        'error': f"HTTP {response.status}: {error_text}"
                    }
        except Exception as e:
            return {
                'request_id': request_id,
                'success': False,
                'error': str(e)
            }
    
    async def run_concurrent_requests():
        """Run 10 concurrent requests"""
        print("🔄 Making 10 concurrent requests...")
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            # Create 10 concurrent tasks
            tasks = [
                make_async_request(session, i, image_base64, base_url, api_key)
                for i in range(50)
            ]
            
            # Wait for all requests to complete
            results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Print individual results
        for result in results:
            if result['success']:
                print(f"   Request {result['request_id']+1}/10 completed in {result['time']:.2f}s")
            else:
                print(f"   Request {result['request_id']+1}/10 failed: {result['error']}")
        
        # Analyze results
        successful_requests = [r for r in results if r['success']]
        failed_requests = [r for r in results if not r['success']]
        
        print(f"📊 Concurrent Test Results:")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Successful requests: {len(successful_requests)}/10")
        print(f"   Failed requests: {len(failed_requests)}/10")
        
        if successful_requests:
            avg_time = sum(r['time'] for r in successful_requests) / len(successful_requests)
            print(f"   Average request time: {avg_time:.2f}s")
            print(f"   Throughput: {len(successful_requests)/total_time:.2f} requests/second")
        
        return len(successful_requests) == 10
    
    # Run the async function
    return asyncio.run(run_concurrent_requests())


if __name__ == "__main__":
    test_endpoints()
    test_authentication()
    test_image_url()
    test_concurrent_queries() 