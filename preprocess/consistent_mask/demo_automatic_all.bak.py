import os

# get dataset_name from command line
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, required=True) # example 3d_ovs
parser.add_argument("--scene", type=str, default=None) # example 3d_ovs
args = parser.parse_args()
dataset_name = args.dataset_name
scene = args.scene

project_path = os.getcwd()
dataset_path = f"{project_path}/dataset/{dataset_name}"
print(dataset_path)

# get folder in dataset_path dir
scene_names = os.listdir(dataset_path)

scene_names = [scene_name for scene_name in scene_names if os.path.isdir(f"{dataset_path}/{scene_name}")]
if scene is not None:
    scene_names = [scene]

scene_names.sort()
print("scene_names:", scene_names)

for scene_name in scene_names:

    img_path = f"{project_path}/output/{dataset_name}/{scene_name}/train/_None_30000/renders"
    if dataset_name == "lerf_ovs" or dataset_name == "lerf_mask" or dataset_name == "360_v2":
        img_path = f"{dataset_path}/{scene_name}/images"
    mask_root = f"{dataset_path}/{scene_name}/mask/video_mask_auto/"
    os.makedirs(mask_root, exist_ok=True)


    cmd = (f"python preprocess/consistent_mask/demo_automatic.py "
          f"--chunk_size 4 --img_path {img_path} "
          f"--amp --temporal_setting semionline --size 480 --output {mask_root} "
          f"--suppress_small_objects --SAM_PRED_IOU_THRESHOLD 0.7")
    print(cmd)
    os.system(cmd) 

    # os.system(f"mv {mask_root}/Annotations {mask_root}/Annotations_color")

    # cmd = (f"python preprocess/consistent_mask/demo_automatic.py "
    #       f"--chunk_size 4 --img_path {img_path} "
    #       f"--amp --temporal_setting semionline --size 480 --output {mask_root} "
    #       f"--use_short_id --suppress_small_objects --SAM_PRED_IOU_THRESHOLD 0.7")
    # print(cmd)
    # os.system(cmd) 
