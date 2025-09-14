# convert RGB mask to a gray mask

import os
import numpy as np
from PIL import Image
import argparse


def rgb2instanceID(rgb):
    return np.dot(rgb[...,:3], [1, 256, 65536]).astype(np.uint32)

def convert_mask(mask_root):

    global_ids = {}
    anno_dir = mask_root + "/Annotations"
    mask_files = os.listdir(anno_dir)
    #print("mask_files:", mask_files)

    #create a new dir to save the mask file
    mask_dir = os.path.join(mask_root, "uni_mask")
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)

    for mask_file in mask_files:
        # get base name of mask_file
        mask_file = os.path.basename(mask_file).split(".")[0]

        mask_path = os.path.join(anno_dir, mask_file + ".png")
        mask = Image.open(mask_path)
        mask = np.array(mask)

        instance_id = rgb2instanceID(mask)
        ids = np.unique(instance_id)

        # for each id in ids, add to global_id if not exist
        for id in ids:
            if id not in global_ids:
                global_ids[id] = len(global_ids)
        
        # convert instance_id to global_id
        for id in global_ids:
            instance_id[instance_id == id] = global_ids[id]
        instance_id = instance_id.astype(np.int16)
        np.save(os.path.join(mask_dir, mask_file + ".npy"), instance_id)

    print("global_ids:", global_ids, len(global_ids))
    print("save to", os.path.join(mask_dir))
    # save global_ids to a file
    with open(os.path.join(mask_root, "global_ids.txt"), "w") as f:
        for id in global_ids:
            f.write(str(global_ids[id]) + "\n")
        
    print(os.path.join(mask_root, "global_ids.txt"), len(global_ids))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True) # example "dataset/{dataset_name}"
    parser.add_argument("--scene", type=str, default=None) # example 3d_ovs
    args = parser.parse_args()
    dataset_path = args.dataset_path
    scene = args.scene

    print(dataset_path)
    scene_names = os.listdir(dataset_path)
    if scene is not None:
        scene_names = [scene]
    scene_names.sort()
    print("scene_names:", scene_names)

    for scene_name in scene_names:
        mask_root = f"{dataset_path}/{scene_name}/mask/video_mask_auto/"
        convert_mask(mask_root)
