import base64
import numpy as np
import requests
from typing import Dict, Optional
import pyvista as pv
from PIL import Image
import yaml
import cv2
from depthany2.viz_utils import get_pcd_colors_from_image, viz_pc, save_pointcloud


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def decode_pointcloud(pointcloud_base64: str, shape: list, pil_img: Image.Image = None) -> pv.PolyData:
    """Decode base64 point cloud data and return PyVista PolyData"""
    points_bytes = base64.b64decode(pointcloud_base64)
    points = np.frombuffer(points_bytes, dtype=np.float32).reshape(shape)
    mesh = pv.PolyData(points.astype(np.float64))
    if pil_img is not None:
        colors = get_pcd_colors_from_image(pil_img)
        mesh['colors'] = colors
    return mesh

def decode_depth_map(depth_base64: str, shape: Optional[list] = None) -> np.ndarray:
    depth_bytes = base64.b64decode(depth_base64)
    depth_flat = np.frombuffer(depth_bytes, dtype=np.float32)
    if shape:
        return depth_flat.reshape(shape)
    return depth_flat

def _get_headers(api_key: Optional[str] = None) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers

def predict_pointcloud(image: Optional[str] = None, image_url: Optional[str] = None,
                      camera_intrinsics: Optional[Dict] = None, base_url: str = "http://localhost:8000",
                      encoder: str = "vitl", dataset: str = "hypersim", model_input_size: int = 518,
                      max_depth: float = 1.0, api_key: Optional[str] = None) -> Dict:
    payload = {
        'camera_intrinsics': camera_intrinsics,
        'encoder': encoder,
        'dataset': dataset,
        'model_input_size': model_input_size,
        'max_depth': max_depth
    }
    if image:
        payload['image'] = image
    elif image_url:
        payload['image_url'] = image_url
    else:
        raise ValueError("Either image or image_url must be provided")
    headers = _get_headers(api_key)
    response = requests.post(f"{base_url}/pc", json=payload, headers=headers)
    response.raise_for_status()
    return response.json()

def predict_metric_depth(image: Optional[str] = None, image_url: Optional[str] = None,
                        base_url: str = "http://localhost:8000",
                        encoder: str = "vitl", dataset: str = "hypersim", model_input_size: int = 518,
                        max_depth: float = 1.0, api_key: Optional[str] = None) -> Dict:
    payload = {
        'encoder': encoder,
        'dataset': dataset,
        'model_input_size': model_input_size,
        'max_depth': max_depth
    }
    if image:
        payload['image'] = image
    elif image_url:
        payload['image_url'] = image_url
    else:
        raise ValueError("Either image or image_url must be provided")
    headers = _get_headers(api_key)
    response = requests.post(f"{base_url}/metric_depth", json=payload, headers=headers)
    response.raise_for_status()
    return response.json()

def predict_relative_depth(image: Optional[str] = None, image_url: Optional[str] = None,
                          base_url: str = "http://localhost:8000",
                          encoder: str = "vitl", dataset: str = "hypersim", model_input_size: int = 518,
                          api_key: Optional[str] = None) -> Dict:
    payload = {
        'encoder': encoder,
        'dataset': dataset,
        'model_input_size': model_input_size
    }
    if image:
        payload['image'] = image
    elif image_url:
        payload['image_url'] = image_url
    else:
        raise ValueError("Either image or image_url must be provided")
    headers = _get_headers(api_key)
    response = requests.post(f"{base_url}/rel_depth", json=payload, headers=headers)
    response.raise_for_status()
    return response.json()

def get_health(base_url: str = "http://localhost:8000", api_key: Optional[str] = None) -> Dict:
    headers = _get_headers(api_key)
    response = requests.get(f"{base_url}/health", headers=headers)
    response.raise_for_status()
    return response.json()

if __name__ == "__main__":
    # --- Config ---
    image_path = "depthany2/server/example/ahg_courtyard.png"
    intrinsics_path = "depthany2/server/example/cam_intrinsics_3072.yaml"
    servers_path = "depthany2/server/client/servers.yaml"
    
    # --- Load camera intrinsics from YAML ---
    def load_intrinsics_from_yaml(yaml_path):
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        K = data['camera_matrix']
        return {
            'fx': float(K[0][0]),
            'fy': float(K[1][1]),
            'cx': float(K[0][2]),
            'cy': float(K[1][2])
        }
    camera_intrinsics = load_intrinsics_from_yaml(intrinsics_path)

    # --- Load server info ---
    with open(servers_path, "r") as f:
        servers = yaml.safe_load(f)
    selected_server = servers["dany2"]

    # --- 1. Health check ---
    health = get_health(base_url=selected_server["base_url"], api_key=selected_server["api_key"])
    print("Health:", health)

    # --- 2. Metric Depth (local image) ---
    encoded = encode_image(image_path)
    metric_result = predict_metric_depth(
        image=encoded,
        base_url=selected_server["base_url"],
        api_key=selected_server["api_key"],
        max_depth=1.0
    )
    print("Metric depth keys:", metric_result.keys())
    if "depth_map" in metric_result:
        depth_map = decode_depth_map(metric_result["depth_map"], metric_result["shape"])
        print(f"Metric depth map shape: {depth_map.shape}")

    # --- 2b. Metric Depth (image URL) ---
    image_url = "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=640&h=480&fit=crop"
    metric_url_result = predict_metric_depth(
        image_url=image_url,
        base_url=selected_server["base_url"],
        api_key=selected_server["api_key"],
        max_depth=1.0
    )
    print("Metric depth (URL) keys:", metric_url_result.keys())
    if "depth_map" in metric_url_result:
        depth_map_url = decode_depth_map(metric_url_result["depth_map"], metric_url_result["shape"])
        print(f"Metric depth map (URL) shape: {depth_map_url.shape}")

    # --- 3. Relative Depth (local image) ---
    rel_result = predict_relative_depth(
        image=encoded,
        base_url=selected_server["base_url"],
        api_key=selected_server["api_key"]
    )
    print("Relative depth keys:", rel_result.keys())
    if "depth_map" in rel_result:
        rel_depth_map = decode_depth_map(rel_result["depth_map"], rel_result["shape"])
        print(f"Relative depth map shape: {rel_depth_map.shape}")

    # --- 4. Pointcloud (local image, with intrinsics as dict) ---
    pil_img = Image.open(image_path).convert('RGB')
    bgr_img = cv2.imread(image_path)
    pc_result = predict_pointcloud(
        image=encoded,
        camera_intrinsics=camera_intrinsics,
        base_url=selected_server["base_url"],
        api_key=selected_server["api_key"],
        max_depth=1.0
    )
    pcd = decode_pointcloud(pc_result["pointcloud"], pc_result["pointcloud_shape"], pil_img=pil_img)
    save_pointcloud(
        points=np.asarray(pcd.points),
        filepath="rest_pointcloud.pcd",
        pil_img_for_color=pil_img
    )
    print("Saved REST API pointcloud as rest_pointcloud.pcd")
    # Two-way viz: direct and from file
    viz_pc(pcd)
    viz_pc("rest_pointcloud.pcd")
    print("Done.")