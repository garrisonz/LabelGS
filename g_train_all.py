import os
import yaml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, required=True) # example 3d_ovs
parser.add_argument("--scene", type=str, default=None)
parser.add_argument("--config_file", type=str, required=True)
parser.add_argument("--start_iteration", type=int, default=None)
parser.add_argument("--exclude_scenes", type=str, default="", help="support multi scene, e.g. room,sofa")
args = parser.parse_args()
dataset_name = args.dataset_name
scene = args.scene
exclude = args.exclude_scenes.split(",")

project_path = os.getcwd()
dataset_path = f"{project_path}/dataset/{dataset_name}"
print(dataset_path)

scene_names = os.listdir(dataset_path)

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
    occlude_flag = "--occlude_flag" if config['occlude_flag'] else ""
    work = config['work']

start_cmd = ""

for scene_name in scene_names:

    output_scene_dir = f"output/{dataset_name}/w{work}_{scene_name}_v{version}"

    if args.start_iteration is not None:
        start_cmd = f"--start_checkpoint {output_scene_dir}/chkpnt{args.start_iteration}.pth"

    cmd = (f"python train.py -s dataset/{dataset_name}/{scene_name} "
           f"-m {output_scene_dir} {eval_cmd} "
           f"--iteration {iteration} --label {start_cmd} {occlude_flag} "
           f"--mask_version {mask_version}  --gpf_flag "
           f"--mask_sample_number {mask_sample_number} ") 
    print(cmd)
    os.system(cmd)
