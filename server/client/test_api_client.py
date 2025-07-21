#!/usr/bin/env python3

import asyncio
import aiohttp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import requests
import time
import yaml
from PIL import Image

from server.client.depth_client import (
    encode_image, decode_pointcloud, decode_depth_map,
    predict_pointcloud, predict_metric_depth, predict_relative_depth, get_health
)
from depthany2.minimal_pts import load_intrinsics_from_yaml


def test_endpoints():
    print("Testing Depth Anything V2 API endpoints...")
    
    with open("server/client/servers.yaml", "r") as f:
        servers = yaml.safe_load(f)
    
    base_url = servers["dany2"]["base_url"]
    api_key = servers["dany2"]["api_key"]
    
    try:
        intrinsics_path = 'server/example/cam_intrinsics_3072.yaml'
        cam_intrinsics = load_intrinsics_from_yaml(intrinsics_path)
        camera_intrinsics = {
            'fx': cam_intrinsics['camera_matrix'][0, 0],
            'fy': cam_intrinsics['camera_matrix'][1, 1],
            'cx': cam_intrinsics['camera_matrix'][0, 2],
            'cy': cam_intrinsics['camera_matrix'][1, 2]
        }
    except FileNotFoundError:
        camera_intrinsics = {'fx': 1000.0, 'fy': 1000.0, 'cx': 640.0, 'cy': 480.0}
    
    try:
        image_path = 'server/example/ahg_courtyard.png'
        image_base64 = encode_image(image_path)
    except FileNotFoundError:
        print("Example image not found. Please provide a valid image path.")
        return False
    
    try:
        health = get_health(base_url, api_key)
        print(f"✅ Health: {health}")
        
        pc_result = predict_pointcloud(
            image=image_base64,
            camera_intrinsics=camera_intrinsics,
            base_url=base_url,
            max_depth=1.0,
            api_key=api_key
        )
        print(f"✅ Pointcloud: shape {pc_result['pointcloud_shape']}")
        
        metric_result = predict_metric_depth(
            image=image_base64,
            base_url=base_url,
            max_depth=1.0,
            api_key=api_key
        )
        print(f"✅ Metric depth: shape {metric_result['shape']}, range [{metric_result['min']:.3f}, {metric_result['max']:.3f}]")
        
        rel_result = predict_relative_depth(
            image=image_base64,
            base_url=base_url,
            api_key=api_key
        )
        print(f"✅ Relative depth: shape {rel_result['shape']}, range [{rel_result['min']:.3f}, {rel_result['max']:.3f}]")
        
        pointcloud = decode_pointcloud(pc_result['pointcloud'], pc_result['pointcloud_shape'], pil_img=Image.open(image_path).convert('RGB'))
        depth_map = decode_depth_map(metric_result['depth_map'], metric_result['shape'])
        
        print(f"Decoded pointcloud: {pointcloud.points.shape}")
        print(f"Decoded depth map: {depth_map.shape}")
        
        matplotlib.use('Agg')
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
    print("\n🔐 Testing API Key Authentication...")
    
    with open("server/client/servers.yaml", "r") as f:
        servers = yaml.safe_load(f)
    
    base_url = servers["dany2"]["base_url"]
    valid_api_key = servers["dany2"]["api_key"]
    
    try:
        health = get_health(base_url)
        print("✅ Health check without API key: SUCCESS")
    except Exception as e:
        print(f"❌ Health check without API key: {e}")
    
    try:
        health = get_health(base_url, api_key=valid_api_key)
        print("✅ Health check with valid API key: SUCCESS")
    except Exception as e:
        print(f"❌ Health check with valid API key: {e}")
    
    try:
        image_path = 'server/example/ahg_courtyard.png'
        image_base64 = encode_image(image_path)
        test_payload = {'image': image_base64, 'max_depth': 1.0}
        
        headers = {"Content-Type": "application/json", "Authorization": "Bearer invalid-key"}
        response = requests.post(f"{base_url}/metric_depth", json=test_payload, headers=headers)
        if response.status_code == 401:
            print("✅ Protected endpoint correctly rejected invalid API key")
        else:
            print(f"❌ Protected endpoint should have rejected invalid API key, got {response.status_code}")
        
        headers_no_auth = {"Content-Type": "application/json"}
        response_no_auth = requests.post(f"{base_url}/metric_depth", json=test_payload, headers=headers_no_auth)
        if response_no_auth.status_code == 401:
            print("✅ Protected endpoint correctly rejected request without API key")
        else:
            print(f"❌ Protected endpoint should have rejected request without API key, got {response_no_auth.status_code}")
            
    except Exception as e:
        print(f"❌ Authentication test error: {e}")


def test_image_url():
    print("\n🌐 Testing Image URL Endpoint...")
    
    with open("server/client/servers.yaml", "r") as f:
        servers = yaml.safe_load(f)
    
    base_url = servers["dany2"]["base_url"]
    api_key = servers["dany2"]["api_key"]
    
    image_url = "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=640&h=480&fit=crop"
    
    try:
        metric_result = predict_metric_depth(
            image_url=image_url,
            base_url=base_url,
            max_depth=1.0,
            api_key=api_key
        )
        print(f"✅ Image URL metric depth: shape {metric_result['shape']}, range [{metric_result['min']:.3f}, {metric_result['max']:.3f}]")
        
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
    print("\n⚡ Testing Concurrent Requests...")
    
    with open("server/client/servers.yaml", "r") as f:
        servers = yaml.safe_load(f)
    
    base_url = servers["dany2"]["base_url"]
    api_key = servers["dany2"]["api_key"]
    
    try:
        image_path = 'server/example/ahg_courtyard.png'
        image_base64 = encode_image(image_path)
    except FileNotFoundError:
        print("Example image not found, skipping concurrent test")
        return False
    
    async def make_async_request(session, request_id, image_base64, base_url, api_key):
        try:
            request_start = time.time()
            
            payload = {'image': image_base64, 'max_depth': 1.0}
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
    
    async def run_concurrent_requests(num_reqs):
        print(f"🔄 Making {num_reqs} concurrent requests...")
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            tasks = [
                make_async_request(session, i, image_base64, base_url, api_key)
                for i in range(num_reqs)
            ]
            results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        for result in results:
            if result['success']:
                print(f"   Request {result['request_id']+1}/{num_reqs} completed in {result['time']:.2f}s")
            else:
                print(f"   Request {result['request_id']+1}/{num_reqs} failed: {result['error']}")
        
        successful_requests = [r for r in results if r['success']]
        failed_requests = [r for r in results if not r['success']]
        
        print(f"📊 Concurrent Test Results:")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Successful requests: {len(successful_requests)}/{num_reqs}")
        print(f"   Failed requests: {len(failed_requests)}/{num_reqs}")
        
        if successful_requests:
            avg_time = sum(r['time'] for r in successful_requests) / len(successful_requests)
            print(f"   Average request time: {avg_time:.2f}s")
            print(f"   Throughput: {len(successful_requests)/total_time:.2f} requests/second")
        
        return len(successful_requests) == num_reqs
    
    return asyncio.run(run_concurrent_requests(50))


if __name__ == "__main__":
    test_endpoints()
    test_authentication()
    test_image_url()
    test_concurrent_queries()