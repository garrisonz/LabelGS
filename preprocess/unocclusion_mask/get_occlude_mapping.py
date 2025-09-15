# label_map is a numpy array with shape (1080, 1440)
# each pixel in label_map is a integer number meaning the label of the pixel
# depth_map is a numpy array with shape (1080, 1440)
# each pixel in depth_map is a float number meaning the depth of the pixel
# for each label region in label_map, find all its neighbor regions which is much closer to the camera at 
# boundary of the region and its neighbor region

import numpy as np
import cv2
import os
from tqdm import tqdm

import argparse

def get_occlude_mapping(scene_path, out_dir, mask_version):

    dataset_name = scene_path.split("/")[1]

    from utils.reader_utils import get_mask_folder

    mask_folder = os.path.join(scene_path, get_mask_folder(mask_version))
    print("mask_version:", mask_version, mask_folder)

    uni_mask_dir = f"{mask_folder}/uni_mask"
    uni_mask_list = os.listdir(uni_mask_dir)
    uni_mask_list.sort()
    uni_mask_name_list = [uni_mask.split(".")[0] for uni_mask in uni_mask_list]

    images_dir = f"{scene_path}/images"
    images_list = os.listdir(images_dir)
    images_list.sort()

    # read rendering_gt_mapping.json
    import json
    rendering_gt_mapping_path = f"{mask_folder}/rendering_gt_mapping.json"
    rendering_gt_mapping = {}
    if os.path.exists(rendering_gt_mapping_path):
        with open(rendering_gt_mapping_path, "r") as f:
            rendering_gt_mapping = json.load(f)
    else:
        print(f"[Warning]{rendering_gt_mapping_path} not found")
        for image_file in images_list:
            image_name = image_file.split(".")[0]
            rendering_gt_mapping[image_name] = image_name

        with open(rendering_gt_mapping_path, "w") as f:
            json.dump(rendering_gt_mapping, f)


    rendering_gt_mapping = {k: v for k, v in rendering_gt_mapping.items() if v in uni_mask_name_list}

    for image_name, mask_name in tqdm(rendering_gt_mapping.items(), desc="occlude analysis"):

        if dataset_name == "lerf_ovs":
            mask_name = image_name

        uni_mask_path = os.path.join(uni_mask_dir, mask_name + ".npy")

        label_map_path = uni_mask_path
        label_map = np.load(label_map_path)

        depth_map_path = f"{scene_path}/depth/predicts/{image_name}.npy"
        depth_map = np.load(depth_map_path)

        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        depth_map = 1 - depth_map

        unique_label = np.unique(label_map)

        #os.makedirs(f"{out_dir}/depth/{image_name}/", exist_ok=True)

        unique_label = unique_label[unique_label != 0]

        occlu_mapping = {}

        for label in unique_label:
            #label = 4
            label_region = (label_map == label)

            # save label_region as image file
            save_label_region = label_region.astype(np.uint8) * 255
            #cv2.imwrite(f"{out_dir}/depth/{image_name}/label_region_{label}.png", save_label_region)

            label_region = label_region.astype(np.uint8)
            erode_label_region = cv2.erode(label_region, np.ones((3, 3), np.uint8), iterations=1)
            dilate_label_region = cv2.dilate(label_region, np.ones((3, 3), np.uint8), iterations=1)

            # save label_region as image file
            save_label_region = erode_label_region.astype(np.uint8) * 255
            #cv2.imwrite(f"{out_dir}/depth/{image_name}/label_region_{label}_dilate.png", save_label_region)

            # get the boundary of label_region by subtracting dilate_label_region from label_region
            in_boundary = label_region - erode_label_region
            out_boundary = dilate_label_region - label_region
            # save boundary as image file
            save_boundary = in_boundary.astype(np.uint8) * 255
            #cv2.imwrite(f"{out_dir}/depth/{image_name}/label_region_{label}_in_boundary.png", save_boundary)

            label_out_boundary = label_map * out_boundary
            depth_out_boundary = depth_map * out_boundary
            unique_label_out_boundary = np.unique(label_out_boundary)
            # remove label 0 from unique_label_out_boundary
            unique_label_out_boundary = unique_label_out_boundary[unique_label_out_boundary != 0]
            occlu_list = []
            for label_out in unique_label_out_boundary:
                #label_out = 8
                out_region = (label_out_boundary == label_out)

                save_out_region = out_region.astype(np.uint8) * 255
                #cv2.imwrite(f"{out_dir}/depth/{image_name}/label_region_{label}_out_{label_out}.png", save_out_region)

                dilate_out_region = cv2.dilate(out_region.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
                in_region = dilate_out_region * in_boundary
                save_in_region = in_region.astype(np.uint8) * 255
                #cv2.imwrite(f"{out_dir}/depth/{image_name}/label_region_{label}_out_{label_out}_inside.png", save_in_region)

                # average depth of pixels in intersection
                depth_in = depth_map * in_region
                depth_in = depth_in[depth_in != 0]
                avg_depth_in = np.mean(depth_in)

                depth_out = depth_out_boundary * out_region
                depth_out = depth_out[depth_out != 0]
                avg_depth_out = np.mean(depth_out)
                #occlu = ""
                if avg_depth_out < avg_depth_in:
                    #occlu = "occlu"
                    #print(f"{label}, {label_out} avg depth. in: {avg_depth_in}, out: {avg_depth_out}, {occlu}")

                    occlu_map = (label_map == label_out)
                    save_occlu_map = occlu_map.astype(np.uint8) * 255
                    #cv2.imwrite(f"{out_dir}/depth/{image_name}/label_region_{label}_out_{label_out}_occlu.png", save_occlu_map)
                    occlu_list.append(int(label_out))

            occlu_mapping[int(label)] = occlu_list

        #print(occlu_mapping)

        # save occlu_mapping as json file
        os.makedirs(f"{mask_folder}/occlude", exist_ok=True) 
        print("save to", f"{mask_folder}/occlude/occlude_mapping_{image_name}.json")
        with open(f"{mask_folder}/occlude/occlude_mapping_{image_name}.json", "w") as f:
            json.dump(occlu_mapping, f)

    print(f"save {mask_folder}/occlude")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True) # f"dataset/{dataset_name}"
    parser.add_argument("--scene", type=str, default=None) # example 3d_ovs
    parser.add_argument("--work", type=str, default="1")
    args = parser.parse_args()
    dataset_path = args.dataset_path
    scene = args.scene
    work = args.work

    dataset_name = dataset_path.split("/")[-1]
    print("dataset_name", dataset_name)

    scene_names = os.listdir(dataset_path)
    if scene is not None:
        scene_names = [scene]
    scene_names.sort()
    print("scene_names:", scene_names)

    version = 1
    mask_version = 2

    for scene_name in scene_names:
        
        scene_path = f"dataset/{dataset_name}/{scene_name}"
        out_dir = f"output/{dataset_name}/w{work}_{scene_name}_v{version}"
        get_occlude_mapping(scene_path, out_dir, mask_version)