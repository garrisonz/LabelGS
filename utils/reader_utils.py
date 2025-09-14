import os

def get_seg_image_list(scene_path):
    seg_path = os.path.join(scene_path, "segmentations")
    seg_image_list = []
    for frame in os.listdir(seg_path):
        if not os.path.isdir(os.path.join(seg_path, frame)):
            continue
        seg_image_list.append(frame)

    seg_image_list.sort()
    return seg_image_list

def get_mask_folder(mask_version : int):
    if mask_version == 2:
        mask_folder = "mask/video_mask_auto"
    elif mask_version == 3:
        # for ["sofa", "table"]:
        mask_folder = "mask/video_mask_auto2"
    elif mask_version == 4:
        mask_folder = "mask/video_mask_auto.deva"
    elif mask_version == 5:
        mask_folder = "mask/merge"
    else:
        mask_folder = "mask/mask_auto"
    
    return mask_folder