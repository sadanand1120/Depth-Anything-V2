import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import subprocess
import os
from PIL import Image
import cv2
import tempfile

def _viz_pc_internal(pc_path, point_size=0.85):
    if not pc_path.endswith(".pcd"):
        raise ValueError("Only .pcd files are supported for visualization.")
    pcd = o3d.io.read_point_cloud(pc_path)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    render_option = vis.get_render_option()
    render_option.point_size = point_size
    vis.add_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    vis.run()
    vis.destroy_window()

def viz_pc(pc_or_path, point_size=0.85):
    """Visualize a point cloud from a .pcd file path or an Open3D PointCloud object using a subprocess."""
    if isinstance(pc_or_path, str):
        subprocess.run(["python3", "-c", f"from depthany2.viz_utils import _viz_pc_internal; _viz_pc_internal('{pc_or_path}', {point_size})"])
    else:
        with tempfile.NamedTemporaryFile(suffix='.pcd', delete=False) as tmp:
            o3d.io.write_point_cloud(tmp.name, pc_or_path)
            subprocess.run(["python3", "-c", f"from depthany2.viz_utils import _viz_pc_internal; _viz_pc_internal('{tmp.name}', {point_size})"])
            os.unlink(tmp.name)

def viz_depth_png(png_path):
    img = plt.imread(png_path)
    plt.imshow(img)
    plt.title("Depth PNG")
    plt.show()

def viz_depth_npy(npy_path):
    arr = np.load(npy_path)
    plt.imshow(arr)
    plt.title("Depth NPY")
    plt.show()


def save_pointcloud(points, filepath, img_for_color=None):
    pcd_path = filepath if filepath.endswith(".pcd") else os.path.splitext(filepath)[0] + ".pcd"
    if img_for_color is not None:
        pil_img = Image.fromarray(cv2.cvtColor(img_for_color, cv2.COLOR_BGR2RGB))
        colors = get_pcd_colors_from_image(pil_img)
        pcd = pcd_from_np(points)
        pcd.colors = colors
    else:
        pcd = pcd_from_np(points)
    o3d.io.write_point_cloud(pcd_path, pcd)
    print(f"Saved point cloud as {pcd_path}")

def get_pcd_colors_from_image(pil_img):
    colors = np.asarray(pil_img).reshape(-1, 3) / 255.0
    return o3d.utility.Vector3dVector(colors)

def pcd_from_np(pc_np, color_rgb_list=None):
    pcd = o3d.geometry.PointCloud()
    xyz = pc_np[:, :3]
    pcd.points = o3d.utility.Vector3dVector(xyz)
    c = [1, 0.647, 0] if color_rgb_list is None else color_rgb_list   # default orange color
    pcd.colors = o3d.utility.Vector3dVector(np.tile(c, (len(xyz), 1)))
    return pcd 