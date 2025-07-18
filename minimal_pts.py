import numpy as np
import cv2
import torch
import yaml
import argparse
import matplotlib.pyplot as plt
import open3d as o3d
from PIL import Image

try:
    from .metric_main import DepthAny2
except:
    from metric_main import DepthAny2


def depth2points(depth_arr_img: np.ndarray, cam_intrinsics_dict):
    FX = cam_intrinsics_dict['camera_matrix'][0, 0]
    FY = cam_intrinsics_dict['camera_matrix'][1, 1]
    CX = cam_intrinsics_dict['camera_matrix'][0, 2]
    CY = cam_intrinsics_dict['camera_matrix'][1, 2]
    x, y = np.meshgrid(np.arange(depth_arr_img.shape[1]), np.arange(depth_arr_img.shape[0]))
    x = (x - CX) / FX
    y = (y - CY) / FY
    points = np.stack((np.multiply(x, depth_arr_img), np.multiply(y, depth_arr_img), depth_arr_img), axis=-1).reshape(-1, 3)
    return points


def predict_depth_from_array(img_array, cam_intrinsics_dict, device=None, encoder='vitl', dataset='hypersim'):
    """
    Predict depth and generate point cloud from a numpy image array.

    Args:
        img_array: Input image as numpy array (HxWx3, BGR format)
        cam_intrinsics_dict: Camera intrinsics dictionary with 'camera_matrix' key
        device: PyTorch device (None for auto-detection)
        encoder: Depth model encoder ('vitl', 'vitb', 'vits')
        dataset: Dataset type ('hypersim', 'vkitti', etc.)

    Returns:
        points: Nx3 numpy array of 3D points
        depth_rel_img: Relative depth image
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
    depth_model = DepthAny2(device=device, model_input_size=518, max_depth=1, encoder=encoder, dataset=dataset)
    depth_rel_img = depth_model.predict(img_array, max_depth=1)
    points = depth2points(depth_rel_img, cam_intrinsics_dict)
    return points, depth_rel_img


def predict_depth_and_pointcloud(img_path, cam_intrinsics_dict, device=None, encoder='vitl', dataset='hypersim', return_image=True):
    """
    Predict depth and generate point cloud from an image.

    Args:
        img_path: Path to input image
        cam_intrinsics_dict: Camera intrinsics dictionary with 'camera_matrix' key
        device: PyTorch device (None for auto-detection)
        encoder: Depth model encoder ('vitl', 'vitb', 'vits')
        dataset: Dataset type ('hypersim', 'vkitti', etc.)
        return_image: Whether to return the original image

    Returns:
        points: Nx3 numpy array of 3D points
        depth_rel_img: Relative depth image
        cv2_img: Original image (if return_image=True)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
    depth_model = DepthAny2(device=device, model_input_size=518, max_depth=1, encoder=encoder, dataset=dataset)
    cv2_img = cv2.imread(img_path)
    depth_rel_img = depth_model.predict(cv2_img, max_depth=1)
    points = depth2points(depth_rel_img, cam_intrinsics_dict)

    if return_image:
        return points, depth_rel_img, cv2_img
    else:
        return points, depth_rel_img


def load_intrinsics_from_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    K = np.array(data['camera_matrix'])
    return {'camera_matrix': K}


def save_pointcloud_bin(points, filename="pointcloud.bin"):
    flat_pc = points.reshape(-1).astype(np.float32)
    flat_pc.tofile(filename)


def get_pcd_colors_from_image(pil_img: Image.Image):
    colors = np.asarray(pil_img).reshape(-1, 3) / 255.0
    return o3d.utility.Vector3dVector(colors)


def pcd_from_np(pc_np, color_rgb_list=None):
    pcd = o3d.geometry.PointCloud()
    xyz = pc_np[:, :3]
    pcd.points = o3d.utility.Vector3dVector(xyz)
    c = [1, 0.647, 0] if color_rgb_list is None else color_rgb_list   # default orange color
    pcd.colors = o3d.utility.Vector3dVector(np.tile(c, (len(xyz), 1)))
    return pcd


def visualize_pcd(pcd, point_size=0.85):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    render_option = vis.get_render_option()
    render_option.point_size = point_size
    vis.add_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    vis.run()
    vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(description="Predict depth and generate pointcloud from image using DepthAny2.")
    parser.add_argument('--image_path', type=str, default='/home/dynamo/AMRL_Research/repos/robot_calib/notrack_bags/paths/poc_cut/images/1742591008774347085.png', help='Path to input image')
    parser.add_argument('--intrinsics_path', type=str, default='/home/dynamo/AMRL_Research/repos/robot_calib/spot/params/cam_intrinsics_3072.yaml', help='Path to camera intrinsics YAML file')
    parser.add_argument('--out', type=str, default='pointcloud.bin', help='Output binary file for pointcloud')
    parser.add_argument('--viz', action='store_true', help='Visualize the pointcloud using Open3D')
    parser.add_argument('--save', action='store_true', help='Save the pointcloud as a binary file')
    args = parser.parse_args()

    cam_intrinsics = load_intrinsics_from_yaml(args.intrinsics_path)
    points, depth_rel_img, cv2_img = predict_depth_and_pointcloud(args.image_path, cam_intrinsics)
    if args.save:
        save_pointcloud_bin(points, args.out)
        print(f"Saved point cloud with {points.shape[0]} points to {args.out}")
    plt.imshow(depth_rel_img)
    plt.title("Predicted Relative Depth")
    plt.show()

    if args.viz:
        pil_img = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
        pcd = pcd_from_np(points)
        pcd.colors = get_pcd_colors_from_image(pil_img)
        visualize_pcd(pcd, point_size=0.75)


if __name__ == "__main__":
    main()
