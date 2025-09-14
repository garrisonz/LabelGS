#
# 
#  Example: python -m  eval.eval_ssim_lpips
#
#

from lpips import lpips  # pip install lpips
from PIL import Image
from utils.image_utils import calculate_ssim, calculate_lpips
import os
import wandb

def eval_scene(scene_path, eval_path, lpips_model):
    ssim_values = []
    lpips_values = []

    seg_path = scene_path + "/segmentations"

    if not os.path.exists(seg_path):
        print(f"{scene_path} has no segmentations. skip.")
        return ssim_values, lpips_values

    print("scene_path:", scene_path)

    scene_name = scene_path.split("/")[-1]

    frames = os.listdir(seg_path)
    frames = sorted(frames)
    frames = frames[1:]


    for frame in frames:
        frame_path = seg_path + "/" + frame
        labels = os.listdir(frame_path)
        labels = sorted(labels)


        for label in labels:
            label_name = label.split(".")[0]

            pred_path = eval_path + f"/test{frame}_{label_name}.png"
            gt_path = eval_path + f"/test{frame}_{label_name}_gt.png"

            if not os.path.exists(pred_path) or not os.path.exists(gt_path):
                print(f"{pred_path} not found")
                continue

            pred_img = Image.open(pred_path)
            gt_img = Image.open(gt_path)

            ssim_value = calculate_ssim(pred_img, gt_img)

            lpips_value = calculate_lpips(lpips_model, pred_img, gt_img) 

            ssim_values.append(ssim_value)
            lpips_values.append(lpips_value)

    if len(ssim_values) == 0:
        print(f"[Error]scene {scene_name} has no ssim values")
        exit()

    return ssim_values, lpips_values


def main(dataset_name, lpips_model, output_version):

    dataset_path = f"dataset/{dataset_name}"
    scenes = os.listdir(dataset_path)
    scenes = sorted(scenes)

    table = wandb.Table(columns=["scene", "ssim_avg", "lpips_avg", "num_samples"])

    ds_ssims = []
    ds_lpips = []

    for scene in scenes:

        scene_path = dataset_path + "/" + scene
        eval_path = f"output/{dataset_name}/auto_{scene}_segEval{output_version}/eval15000"

        ssim_values, lpips_values = eval_scene(scene_path, eval_path, lpips_model)
        if len(ssim_values) == 0:
            continue

        ds_ssims.extend(ssim_values)
        ds_lpips.extend(lpips_values)

        print(f"scene {scene}. SSIM AVG:", sum(ssim_values) / len(ssim_values))
        print(f"scene {scene}. LPIPS AVG:", sum(lpips_values) / len(lpips_values))

        table.add_data(scene, sum(ssim_values) / len(ssim_values), sum(lpips_values) / len(lpips_values), len(ssim_values))

        # wandb.log({
        #     f"{scene}/ssim_avg": sum(ssim_values) / len(ssim_values),
        #     f"{scene}/lpips_avg": sum(lpips_values) / len(lpips_values),
        #     f"{scene}/num_samples": len(ssim_values)
        # })
    
    table.add_data("Average", sum(ds_ssims) / len(ds_ssims), sum(ds_lpips) / len(ds_lpips), len(ds_ssims))

    wandb.log({"ssim_lpips": table})


if __name__ == "__main__":

    lpips_model = lpips.LPIPS(net="vgg").eval().to("cuda")
    dataset_name = "nerf_llff_data"
    output_version = 3.2
    sub_version = 1

    wandb.init(project="LabelGS", name=f"{dataset_name}_v{output_version}_sub{sub_version}"
        ,config={
        "dataset": dataset_name, 
        "version": output_version
        })

    main(dataset_name, lpips_model, output_version)

    wandb.finish()