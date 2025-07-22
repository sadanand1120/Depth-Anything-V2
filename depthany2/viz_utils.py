import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import os
from PIL import Image
import cv2
import tempfile

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("Open3D not available. PCD file support disabled. Install with: pip install open3d>=0.16.0")

def _load_pcd_via_open3d(pcd_path):
    if not HAS_OPEN3D:
        raise ImportError("Open3D required for PCD file support.")
    o3d_pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(o3d_pcd.points)
    mesh = pv.PolyData(points)
    if o3d_pcd.has_colors():
        mesh['colors'] = np.asarray(o3d_pcd.colors)
    return mesh

def _save_pcd_via_open3d(pv_mesh, pcd_path):
    if not HAS_OPEN3D:
        raise ImportError("Open3D required for PCD file support.")
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(pv_mesh.points.astype(np.float64))
    if 'colors' in pv_mesh.point_data:
        o3d_pcd.colors = o3d.utility.Vector3dVector(pv_mesh['colors'].astype(np.float64))
    o3d.io.write_point_cloud(pcd_path, o3d_pcd)

def viz_pc(pc_or_path, point_size=0.85):
    if isinstance(pc_or_path, str):
        if pc_or_path.endswith('.pcd'):
            mesh = _load_pcd_via_open3d(pc_or_path)
        else:
            mesh = pv.read(pc_or_path)
    else:
        if hasattr(pc_or_path, 'points'):
            mesh = pc_or_path
        else:
            mesh = pv.PolyData(pc_or_path)
    plotter = pv.Plotter()
    if 'colors' in mesh.point_data:
        plotter.add_mesh(mesh, style='points', point_size=point_size, scalars='colors', rgb=True)
    else:
        plotter.add_mesh(mesh, style='points', point_size=point_size, color='orange')
    plotter.show()

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

def save_pointcloud(points, filepath, pil_img_for_color=None):
    base_path = os.path.splitext(filepath)[0]
    ext = os.path.splitext(filepath)[1].lower()
    mesh = pv.PolyData(points)
    if pil_img_for_color is not None:
        colors = get_pcd_colors_from_image(pil_img_for_color)
        mesh['colors'] = colors
    if ext == '.pcd':
        _save_pcd_via_open3d(mesh, filepath)
    else:
        if ext not in ['.ply', '.vtk', '.vtp', '.stl']:
            filepath = base_path + '.ply'
        mesh.save(filepath)

def get_pcd_colors_from_image(pil_img):
    """Extract normalized RGB colors from a PIL.Image.Image (RGB)"""
    colors = np.asarray(pil_img).reshape(-1, 3) / 255.0
    return colors

def pcd_from_np(pc_np, color_rgb_list=None):
    mesh = pv.PolyData(pc_np[:, :3])
    if color_rgb_list is not None:
        if len(color_rgb_list) == 3:
            colors = np.tile(color_rgb_list, (len(pc_np), 1))
        else:
            colors = np.array(color_rgb_list)
        mesh['colors'] = colors
    else:
        # Default orange color
        default_color = [1, 0.647, 0]
        mesh['colors'] = np.tile(default_color, (len(pc_np), 1))
    return mesh

def load_point_cloud(filepath):
    if filepath.endswith('.pcd'):
        return _load_pcd_via_open3d(filepath)
    else:
        return pv.read(filepath)

def convert_pcd_to_ply(pcd_path, ply_path=None):
    if ply_path is None:
        ply_path = os.path.splitext(pcd_path)[0] + '.ply'
    mesh = _load_pcd_via_open3d(pcd_path)
    mesh.save(ply_path)
    print(f"Converted {pcd_path} to {ply_path}")
    return ply_path 