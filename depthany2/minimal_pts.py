import numpy as np
import cv2
import torch
import yaml
import os
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from depthany2.metric_main import DepthAny2
from depthany2.viz_utils import viz_pc, save_pointcloud, get_pcd_colors_from_image, pcd_from_np


class DepthPredictor:
    def __init__(self, device=None, encoder='vitl', dataset='hypersim', model_input_size=518, max_depth=1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.model = DepthAny2(device=self.device, model_input_size=model_input_size, max_depth=max_depth, 
                              encoder=encoder, dataset=dataset)
        self.max_depth = max_depth
    
    def predict(self, img_input, cam_intrinsics_dict, return_image=False):
        """Predict depth and generate point cloud from image (file path or numpy array)"""
        if isinstance(img_input, str):
            img_array = cv2.imread(img_input)
            original_img = img_array
        else:
            img_array = img_input
            original_img = img_array
            
        depth_rel_img = self.model.predict(img_array, max_depth=self.max_depth)
        points = depth2points(depth_rel_img, cam_intrinsics_dict)
        
        if return_image:
            return points, depth_rel_img, original_img
        return points, depth_rel_img


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


def predict_depth_and_points(img_input, cam_intrinsics_dict, device=None, encoder='vitl', dataset='hypersim', 
                           model_input_size=518, max_depth=1, return_image=False):
    """Predict depth and generate point cloud from image (file path or numpy array)"""
    predictor = DepthPredictor(device=device, encoder=encoder, dataset=dataset, 
                              model_input_size=model_input_size, max_depth=max_depth)
    return predictor.predict(img_input, cam_intrinsics_dict, return_image=return_image)


def load_intrinsics_from_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    K = np.array(data['camera_matrix'])
    return {'camera_matrix': K}


def main():
    parser = argparse.ArgumentParser(description="Predict depth and generate pointcloud from image using DepthAny2.")
    parser.add_argument('--image_path', type=str, default='/home/dynamo/AMRL_Research/repos/robot_calib/notrack_bags/paths/poc_cut/images/1742591008774347085.png', help='Path to input image')
    parser.add_argument('--intrinsics_path', type=str, default='/home/dynamo/AMRL_Research/repos/robot_calib/spot/params/cam_intrinsics_3072.yaml', help='Path to camera intrinsics YAML file')
    parser.add_argument('--pc_outpath', type=str, default='pointcloud.pcd', help='Output filepath for pointcloud (auto-detects format from extension, always saves as .pcd)')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vitl', 'vitb', 'vits'], help='Model encoder')
    parser.add_argument('--dataset', type=str, default='hypersim', help='Dataset type')
    parser.add_argument('--model_size', type=int, default=518, help='Model input size')
    parser.add_argument('--max_depth', type=float, default=1.0, help='Maximum depth value')
    parser.add_argument('--viz', action='store_true', help='Visualize the pointcloud using Open3D')
    parser.add_argument('--save', action='store_true', help='Save the pointcloud')
    parser.add_argument('--no_depth_plot', action='store_true', help='Skip depth visualization plot')
    args = parser.parse_args()

    cam_intrinsics = load_intrinsics_from_yaml(args.intrinsics_path)
    points, depth_rel_img, cv2_img = predict_depth_and_points(
        args.image_path, cam_intrinsics, encoder=args.encoder, dataset=args.dataset,
        model_input_size=args.model_size, max_depth=args.max_depth, return_image=True
    )
    
    if args.save:
        save_pointcloud(points, args.pc_outpath, img_for_color=cv2_img)
        np.save(os.path.splitext(args.pc_outpath)[0] + '_depth.npy', depth_rel_img)
        plt.imsave(os.path.splitext(args.pc_outpath)[0] + '_depth.png', depth_rel_img)
        print(f"Saved depth npy and png to {os.path.splitext(args.pc_outpath)[0]}_depth.npy/.png")
    
    if not args.no_depth_plot:
        plt.imshow(depth_rel_img)
        plt.title("Predicted Relative Depth")
        plt.show()

    if args.viz:
        pil_img = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
        pcd = pcd_from_np(points)
        pcd.colors = get_pcd_colors_from_image(pil_img)
        viz_pc(pcd, point_size=0.75)


if __name__ == "__main__":
    main()
