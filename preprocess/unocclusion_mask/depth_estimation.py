import argparse
import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch

import sys
sys.path.append("/home/zhangyupeng/w/3drecon/Depth-Anything-V2")

from depth_anything_v2.dpt import DepthAnythingV2

def depth_estimation(img_path: str, input_size, outdir, encoder, pred_only, grayscale):
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**model_configs[encoder])
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    if os.path.isfile(img_path):
        if img_path.endswith('txt'):
            with open(img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [img_path]
    else:
        filenames = glob.glob(os.path.join(img_path, '**/*'), recursive=True)
    
    os.makedirs(outdir, exist_ok=True)
    
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    
    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')
        
        raw_image = cv2.imread(filename)
        
        depth = depth_anything.infer_image(raw_image, input_size)

        # save depth as numpy array
        os.makedirs(os.path.join(outdir, "predicts"), exist_ok=True)
        np.save(os.path.join(outdir, "predicts", os.path.splitext(os.path.basename(filename))[0] + '.npy'), depth)
        
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        
        if grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        
        os.makedirs(os.path.join(outdir, "visualizations"), exist_ok=True)
        if pred_only:
            cv2.imwrite(os.path.join(outdir, "visualizations", os.path.splitext(os.path.basename(filename))[0] + '.png'), depth)
        else:
            split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
            combined_result = cv2.hconcat([raw_image, split_region, depth])
            
            cv2.imwrite(os.path.join(outdir, "visualizations", os.path.splitext(os.path.basename(filename))[0] + '.png'), combined_result)



def scene_depth_estimation(img_path, outdir):

    input_size = 518                 # 归一尺寸 (默认518)
    encoder = "vitl"                 # 编码器类型 ['vits', 'vitb', 'vitl', 'vitg']
    pred_only = True                # 是否仅显示预测结果
    grayscale = True                # 是否禁用彩色调色板

    # 调用函数
    depth_estimation(
        img_path=img_path,
        input_size=input_size,
        outdir=outdir,
        encoder=encoder,
        pred_only=pred_only,
        grayscale=grayscale
    )

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True) # example "dataset/{dataset_name}"
    parser.add_argument("--scene", type=str, default=None) # example 3d_ovs
    args = parser.parse_args()
    dataset_path = args.dataset_path
    scene = args.scene

    print(dataset_path)

    scene_names = os.listdir(dataset_path)
    scene_names.sort()
    scene_names = [scene_name for scene_name in scene_names if os.path.isdir(f"{dataset_path}/{scene_name}")]

    if scene is not None:
        scene_names = [scene]

    print("scene_names:", scene_names)

    for scene_name in scene_names:

        img_path = f"{dataset_path}/{scene_name}/images/"
        outdir = f"{dataset_path}/{scene_name}/depth/"

        scene_depth_estimation(img_path, outdir)
    