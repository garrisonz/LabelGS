import os
import cv2
import numpy as np


def downsample_image(source_folder, output_folder, max_height, is_mask=False):
    img_dir = source_folder
    output_folder = output_folder
    os.makedirs(output_folder, exist_ok=True)

    #remove img_dir_8
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    print(img_dir)
    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)
        if not img_name.endswith(".jpg") and not img_name.endswith(".png") and not img_name.endswith(".JPG"):
            continue
        img = cv2.imread(img_path)

        image = img
        orig_w, orig_h = image.shape[1], image.shape[0]
        if orig_h > max_height:
            global_down = orig_h / max_height
        else:
            global_down = 1
        scale = float(global_down)
        resolution = (int( orig_w  / scale), int(orig_h / scale))
        image = cv2.resize(image, resolution)
        img = image
        img = img.astype(np.uint8)

        if is_mask:
            img = img > 128
            img = img.astype(np.uint8) * 255

        cv2.imwrite(os.path.join(output_folder, img_name), img)
        print(f"downsampled {img_name} to {output_folder} {img.shape}")


def downsample_scene(scene_path, max_height=1080):
    images_path = scene_path + "/images"
    input_path = scene_path + "/images_input"
    if not os.path.exists(input_path):
        os.rename(images_path, input_path)

    downsample_image(input_path, images_path, max_height, is_mask=False)

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path', type=str, required=True) # "dataset/{dataset_name}"
    parser.add_argument("--max_height", type=int, default=1080)
    args = parser.parse_args()
    dataset_path = args.dataset_path
    max_height = args.max_height

    scene_names = os.listdir(dataset_path)
    scene_names.sort()

    print("scene_names:", scene_names)

    for scene_name in scene_names:
        scene_path = f"{dataset_path}/{scene_name}"
        downsample_scene(scene_path, max_height)