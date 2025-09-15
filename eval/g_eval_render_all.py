import os
import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, required=True) # example 3d_ovs
parser.add_argument("--scene", type=str, default=None)
parser.add_argument("--config_file", type=str, required=True)
parser.add_argument("--white_background", action="store_true", default=False)
parser.add_argument("--exclude_scenes", type=str, default="", help="support multi scene, e.g. room,sofa")

args = parser.parse_args()
dataset_name = args.dataset_name
white_background = args.white_background
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


with open(args.config_file, 'r') as f:
    config = yaml.safe_load(f)
    iteration = config['iteration']
    version = config['version']
    mask_version = config['mask_version']
    eval_cmd = "--eval" if config['eval_flag'] else ""
    gpf_flag =  "--gpf_flag" if config['gpf_flag'] else ""
    mask_sample_number = config['mask_sample_number']
    work = config['work']
    loaded_iter = config['iteration']

if white_background:
    white_background = "--white_background"
else:
    white_background = ""

for scene_name in scene_names:

    cmd = (f"python -m eval.eval_render -m output/{dataset_name}/w{work}_{scene_name}_v{version} "
           f"--loaded_iter {loaded_iter} {white_background} "
           f"--dataset_name {dataset_name} "
           f"--scene_name {scene_name} "
           f"--mask_version {mask_version} "
           f"--version {version}")
    print(cmd)
    os.system(cmd)
    print("")


