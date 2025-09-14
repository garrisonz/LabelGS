import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, required=True) # example 3d_ovs
parser.add_argument("--white_background", action="store_true", default=False)
parser.add_argument("--nas_output", action="store_true", default=False)
parser.add_argument("--scene", type=str, default=None)
parser.add_argument("--exclude_scenes", type=str, default="", help="support multi scene, e.g. room,sofa") # example 3d_ovs
args = parser.parse_args()
dataset_name = args.dataset_name
white_background = args.white_background
nas_output = args.nas_output
scene = args.scene
exclude = args.exclude_scenes.split(",") 

project_path = os.getcwd()
dataset_path = f"{project_path}/dataset/{dataset_name}"
print(dataset_path)

# get folder in dataset_path dir
scene_names = os.listdir(dataset_path)
scene_names = [scene_name for scene_name in scene_names if os.path.isdir(f"{dataset_path}/{scene_name}/segmentations")]

if exclude is not None:
    scene_names = [scene_name for scene_name in scene_names if scene_name not in exclude]
if scene is not None:
    scene_names = [scene]
scene_names.sort()
print("scene_names:", scene_names)

loaded_iter = 15000
version = 3.2
mask_version = 3

if white_background:
    white_background = "--white_background"
else:
    white_background = ""


for scene_name in scene_names:

    cmd = (f"python -m eval.eval_psnr_iou -m output/{dataset_name}/auto_{scene_name}_segEval{version} --loaded_iter {loaded_iter} {white_background} --dataset_name {dataset_name} --scene_name {scene_name} --mask_version {mask_version} --version {version}")
    print(cmd)
    os.system(cmd)
    print("")


