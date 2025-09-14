import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, required=True) # example 3d_ovs
parser.add_argument("--eval", action="store_true", default=False)
parser.add_argument('--mask_sample_number', type=int, default=10)
parser.add_argument("--scene", type=str, default=None)
parser.add_argument("--start_iteration", type=int, default=None)
parser.add_argument("--exclude_scenes", type=str, default="", help="support multi scene, e.g. room,sofa") # example 3d_ovs
args = parser.parse_args()
dataset_name = args.dataset_name
eval = args.eval
scene = args.scene
exclude = args.exclude_scenes.split(",")

dataset_path = f"/home/zhangyupeng/w/3drecon/LabelGS/dataset/{dataset_name}"
print(dataset_path)

# get folder in dataset_path dir
scene_names = os.listdir(dataset_path)
scene_names = [scene_name for scene_name in scene_names if os.path.isdir(f"{dataset_path}/{scene_name}")]
if dataset_name == "lerf_ovs":
    scene_names = [scene_name for scene_name in scene_names if "label" != scene_name]

if dataset_name == "360_v2":
    scene_names = [scene_name for scene_name in scene_names if os.path.isdir(f"{dataset_path}/{scene_name}/segmentations")]
if exclude is not None:
    scene_names = [scene_name for scene_name in scene_names if scene_name not in exclude]
if scene is not None:
    scene_names = [scene]
scene_names.sort()

print("scene_names:", scene_names)

start_cmd = ""
eval_cmd = ""

iteration = 15000
version = 3.2
mask_version = 2

gpf_flag_str = "--gpf_flag"
if version == 3.2:
    gpf_flag_str = ""


if eval:
    eval_cmd = "--eval"

for scene_name in scene_names:

    output_scene_dir = f"output/{dataset_name}/auto_{scene_name}_segEval{version}"

    if args.start_iteration is not None:
        start_cmd = f"--start_checkpoint {output_scene_dir}/chkpnt{args.start_iteration}.pth"

    cmd = (f"python train.py -s dataset/{dataset_name}/{scene_name} "
           f"-m {output_scene_dir} {eval_cmd} "
           f"--iteration {iteration} --label {start_cmd} --occlude_flag "
           f"--mask_version {mask_version} {gpf_flag_str} "
           f"--mask_sample_number {args.mask_sample_number} "
           f"--version {version} ") 
    print(cmd)
    os.system(cmd)

