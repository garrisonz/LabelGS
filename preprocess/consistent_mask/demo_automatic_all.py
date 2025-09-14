import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, required=True) # example 3d_ovs
parser.add_argument("--scene", type=str, default=None) # example 3d_ovs
parser.add_argument("--densify_views", action="store_true", default=False) # 3d_ovs 有需要
args = parser.parse_args()

dataset_name = args.dataset_name
scene = args.scene
densify_views = args.densify_views

project_path = os.getcwd()
dataset_path = f"{project_path}/dataset/{dataset_name}"

scene_names = os.listdir(dataset_path)

scene_names = [scene_name for scene_name in scene_names if os.path.isdir(f"{dataset_path}/{scene_name}")]
if scene is not None:
    scene_names = [scene]
scene_names.sort()
print("scene_names:", scene_names)

for scene_name in scene_names:

    if densify_views:
        img_path = f"{project_path}/output/{dataset_name}/{scene_name}/train/_None_30000/renders"
    else:
        img_path = f"{dataset_path}/{scene_name}/images"

    mask_root = f"{dataset_path}/{scene_name}/mask/video_mask_auto/"
    os.makedirs(mask_root, exist_ok=True)

    cmd = (f"python preprocess/consistent_mask/demo_automatic.py "
          f"--chunk_size 4 --img_path {img_path} "
          f"--amp --temporal_setting semionline --size 480 --output {mask_root} "
          f"--suppress_small_objects --SAM_PRED_IOU_THRESHOLD 0.7")
    print(cmd)
    os.system(cmd) 
