from turtle import pos
import numpy as np
import test
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from scene import cameras
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene.cameras import Camera
import logging

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import numpy as np
from scipy.spatial.transform import Slerp, Rotation

def compute_angular_distance(cam1, cam2):
    """计算两个相机视角之间的旋转角度差（弧度）"""
    # 将旋转矩阵转换为 scipy 的 Rotation 对象
    R1 = Rotation.from_matrix(cam1.R)
    R2 = Rotation.from_matrix(cam2.R)
    
    # 计算相对旋转并提取角度
    rel_rotation = R1.inv() * R2
    return rel_rotation.magnitude()

def interpolate_cameras(cam1, cam2, t):
    """在两个相机之间插值（t 范围 [0,1]）"""
    # 旋转插值（使用 Slerp）
    rotations = Rotation.concatenate([
        Rotation.from_matrix(cam1.R), 
        Rotation.from_matrix(cam2.R)
    ])
    slerp = Slerp([0, 1], rotations)
    R = slerp(t).as_matrix()
    
    # 平移插值（线性）
    T = (1 - t) * cam1.T + t * cam2.T
    
    # FoV 插值（线性）
    FovX = (1 - t) * cam1.FoVx + t * cam2.FoVx
    FovY = (1 - t) * cam1.FoVy + t * cam2.FoVy

    return Camera(colmap_id=cam1.uid, R=R, T=T, 
                  FoVx=FovX, FoVy=FovY, 
                  image=cam1.original_image,
                  gt_alpha_mask=None,
                  image_name=f"interp_{cam1.image_name}_{cam2.image_name}_{t:.2f}",  
                  mask_list=[], 
                  sdf_list=None, occlude_mapping=None, uid=None)


def find_max_distance_cameras(cameras):
    # 遍历所有相机对，找到角度差最大的两个相机
    max_distance = -1
    camA = None
    camB = None
    for i in range(len(cameras)):
        for j in range(i+1, len(cameras)):
            dist = compute_angular_distance(cameras[i], cameras[j])
            if dist > max_distance:
                max_distance = dist
                camA, camB = (cameras[i], cameras[j])
    return camA, camB

def get_cameras_from_name(cameras, name):
    for cam in cameras:
        if cam.image_name == name:
            return cam
    
    print("[Error]Camera not found", name)
    return None

def get_labels_from_mask(cameras, image_name, gaussians, pipeline, background, args, mask):
    view = get_cameras_from_name(cameras, image_name)
    render_pkg = render(view, gaussians, pipeline, background, args)
    alpha_id_map = render_pkg["alpha_id_map"].cpu()

    alpha_id_map = alpha_id_map.type(torch.long)
    label_map = gaussians.label[alpha_id_map].cpu().numpy()
    label_map_cnt = np.unique(label_map, return_counts=True)
    label_map_cnt = dict(zip(label_map_cnt[0], label_map_cnt[1]))

    mask_label_cnt = np.unique(label_map * mask, return_counts=True)
    mask_label_cnt = dict(zip(mask_label_cnt[0], mask_label_cnt[1]))
    mask_label_cnt.pop(0, None)
    mask_label_cnt.pop(-1, None)
    valid_labels = [k for k, v in mask_label_cnt.items() if v > label_map_cnt[k] * 0.5]

    valid_labels = torch.tensor(valid_labels, dtype=torch.int32, device="cuda")

    return valid_labels

def render_by_mask_prompt(dataset : ModelParams, pipeline : PipelineParams, args):
    dataset.eval = True

    scene_name = args.scene_name
    print(scene_name)
    dataset_name = args.dataset_name
    print(dataset_name)

    eval_path = args.model_path.replace("output", "result", 1) + f"/eval{args.loaded_iter}"
    makedirs(eval_path, exist_ok=True)

    mask_view_name = "00"
    mask_path = f"dataset/{dataset_name}/{scene_name}/segmentations_w_bg/{mask_view_name}/weaving basket.png"
    mask = torchvision.io.read_image(mask_path).to("cuda").float()
    mask = mask[0, ...] > 0
    mask = np.array(mask.cpu())

    with torch.no_grad():
        # read gaussian model
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, shuffle=False, version=args.version, mask_version=args.mask_version)
        checkpoint = os.path.join(args.model_path, f'chkpnt{args.loaded_iter}.pth')
        (model_params, _) = torch.load(checkpoint)
        gaussians.restore(model_params, args, mode='test')

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        train_cameras = scene.getTrainCameras()
        test_cameras = scene.getTestCameras()
        cameras = train_cameras + test_cameras

        camA = get_cameras_from_name(cameras, "13")
        camB = get_cameras_from_name(cameras, "30")

        interpolated_cams = [interpolate_cameras(camA, camB, t) for t in np.linspace(0, 1, 10)]

        # 获得 valid_labels
        valid_labels = get_labels_from_mask(cameras, mask_view_name, gaussians, pipeline, background, args, mask)

        for view in interpolated_cams:
            print(view.T, view.image_name)
            render_pkg = render(view, gaussians, pipeline, background, args)
            torchvision.utils.save_image(render_pkg["render"], f"{eval_path}/{view.image_name}.png")

            render_pkg2 = render(view, gaussians, pipeline, background, args, label_id=valid_labels)
            torchvision.utils.save_image(render_pkg2["render"], f"{eval_path}/{view.image_name}_seg.png")




            


        


if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--loaded_iter", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--dataset_name", type=str, required=True) # example 3d_ovs
    parser.add_argument("--scene_name", type=str, required=True) # example 3d_ovs
    parser.add_argument("--mask_version", type=int, required=True)
    parser.add_argument("--version", type=str, required=True)

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)

    log_file_name = f"{args.model_path}/eval.log"
    logging.basicConfig(filename=log_file_name, level=logging.INFO)

    render_by_mask_prompt(model.extract(args), pipeline.extract(args), args)


