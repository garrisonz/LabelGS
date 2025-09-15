import numpy as np
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.image_utils import psnr
import logging
import cv2
from PIL import Image


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

#def gaussian_render(dataset : ModelParams, pipeline : PipelineParams, args, save_path : str, white_background=False):
def gaussian_render(dataset, pipeline, args, save_path : str, white_background=False):

    # Render all views
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, shuffle=False, version=args.version, mask_version=args.mask_version)
    checkpoint = os.path.join(args.model_path, f'chkpnt{args.loaded_iter}.pth')
    (model_params, _) = torch.load(checkpoint)
    gaussians.restore(model_params, args, mode='test')

    train_cameras = scene.getTrainCameras()
    test_cameras = scene.getTestCameras()
    
    bg_color = [1,1,1] if white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    all_cameras = train_cameras + test_cameras
    os.makedirs(f"{save_path}/full_scene", exist_ok=True)
    render_all_views(gaussians, background, all_cameras, f"{save_path}/full_scene", tag="pre")

    # get target label
    del_path = f"dataset/occlusion/clothing/edition/delete"
    frames = os.listdir(del_path)
    for frame in frames:
        mask_imgs = os.listdir(f"{del_path}/{frame}")
        view = get_view(all_cameras, frame)
        for mask in mask_imgs:
            mask_path = f"{del_path}/{frame}/{mask}"
            binary_mask = load_binary_mask(mask_path)
            label_ids = get_target_label(gaussians, background, view, binary_mask)
            obj_name = mask.split(".")[0]

            os.makedirs(f"{save_path}/{obj_name}", exist_ok=True)

            # segment object
            label_ids = torch.tensor(label_ids, dtype=torch.int32, device="cuda")
            render_all_views(gaussians, background, all_cameras, f"{save_path}/{obj_name}", tag=f"seg", label_id=label_ids)

            # delete object
            all_labels = torch.unique(gaussians.label).to(device="cuda")
            mask = ~torch.isin(all_labels, label_ids)
            other_labels = all_labels[mask]
            non_neg_mask = other_labels != -1
            non_neg = other_labels[non_neg_mask]
            neg_ones = other_labels[~non_neg_mask]
            other_labels_sorted = torch.cat([non_neg, neg_ones])
            render_all_views(gaussians, background, all_cameras, f"{save_path}/{obj_name}", tag=f"del", label_id=other_labels_sorted)

            


def get_view(all_camers, frame_name):
    for c in all_camers:
        if c.image_name == frame_name:
            return c


def load_binary_mask(mask_path, threshold=128):
    img = Image.open(mask_path).convert("L")
    mask_array = np.array(img)
    binary_mask = (mask_array >= threshold).astype(np.uint8)
    return binary_mask

def render_all_views(gaussians, background, views, save_path, tag, label_id=None):

    for view in tqdm(views):
        render_pkg = render(view, gaussians, pipeline, background, args, label_id=label_id)
        image_name = view.image_name

        torchvision.utils.save_image(render_pkg["render"],  save_path+f"/{tag}_{image_name}.png")

def get_target_label(gaussians, background, view, mask):
    render_pkg = render(view, gaussians, pipeline, background)
    alpha_id_map = render_pkg["alpha_id_map"].type(torch.long)
    label_map = gaussians.label[alpha_id_map].cpu().numpy()

    label_map_cnt = np.unique(label_map, return_counts=True)
    label_map_cnt = dict(zip(label_map_cnt[0], label_map_cnt[1]))
    print("label_map_cnt:", label_map_cnt)

    mask_label_cnt = np.unique(label_map * mask, return_counts=True)
    mask_label_cnt = dict(zip(mask_label_cnt[0], mask_label_cnt[1]))
    mask_label_cnt.pop(0, None)
    mask_label_cnt.pop(-1, None)
    valid_label = [k for k, v in mask_label_cnt.items() if v > label_map_cnt[k] * 0.5]

    return valid_label
        

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
    parser.add_argument("--work", type=str, default="1")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)

    log_file_name = f"{args.model_path}/eval.log"
    logging.basicConfig(filename=log_file_name, level=logging.INFO)

    dataset = model.extract(args)
    pipeline = pipeline.extract(args)
    dataset_name = args.dataset_name
    scene_name = args.scene_name
    version = args.version
    iteration = args.loaded_iter
    work = args.work

    save_path = f"result/{dataset_name}/w{work}_{scene_name}_v{version}/render{iteration}"
    makedirs(save_path, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)

    gaussian_render(dataset, pipeline, args, save_path)


