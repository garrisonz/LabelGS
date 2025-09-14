import os
import yaml
import argparse
import csv


# 读取 CSV 文件并构建字典
# 一个场景的汇总数据，因此只有一行数据
def read_csv_to_dict(file_path):
    data_dict = []
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data_dict.append({
                'scene': str(row['scene']),
                'frame_ssim_avg': float(row['frame_ssim_avg']),
                'frame_lpips_avg': float(row['frame_lpips_avg']),
                'frame_psnr_avg': float(row['frame_psnr_avg']),
                'frame_num': int(row['frame_num']),
                'obj_iou_avg': float(row['obj_iou_avg']),
                'obj_ssim_avg': float(row['obj_ssim_avg']),
                'obj_lpips_avg': float(row['obj_lpips_avg']),
                'obj_psnr_avg': float(row['obj_psnr_avg']),
                'obj_num': int(row['obj_num'])
            })
    #print(f"Read {file_path} to dict: {data_dict}")
    return data_dict[0]

def cal_dataset_metrics(dataset_name, scene_names, version, iteration, work):
    metrics = {
        "frame_psnr": [],
        "frame_ssim": [],
        "frame_lpips": [],
        "frame_num": [],
        "obj_iou": [],
        "obj_psnr": [],
        "obj_ssim": [],
        "obj_lpips": [],
        "obj_num": [],
    }

    rows = []

    for scene_name in scene_names:
        result_scene_dir = f"result/{dataset_name}/w{work}_{scene_name}_v{version}"
        metric_path = f"{result_scene_dir}/eval{iteration}/metrics.csv"

        if not os.path.exists(metric_path):
            print(f"metrics.csv not found: {metric_path}")
            continue

        data_dict = read_csv_to_dict(metric_path)

        row = [
            scene_name,
            data_dict["frame_ssim_avg"],
            data_dict["frame_lpips_avg"],
            data_dict["frame_psnr_avg"],
            data_dict["frame_num"],
            data_dict["obj_iou_avg"],
            data_dict["obj_ssim_avg"],
            data_dict["obj_lpips_avg"],
            data_dict["obj_psnr_avg"],
            data_dict["obj_num"]
        ]
        rows.append(row)

        metrics["frame_psnr"].append(data_dict["frame_psnr_avg"])
        metrics["frame_ssim"].append(data_dict["frame_ssim_avg"])
        metrics["frame_lpips"].append(data_dict["frame_lpips_avg"])
        metrics["frame_num"].append(data_dict["frame_num"])
        metrics["obj_iou"].append(data_dict["obj_iou_avg"])
        metrics["obj_psnr"].append(data_dict["obj_psnr_avg"])
        metrics["obj_ssim"].append(data_dict["obj_ssim_avg"])
        metrics["obj_lpips"].append(data_dict["obj_lpips_avg"])
        metrics["obj_num"].append(data_dict["obj_num"])

    
    # 定义 CSV 列名
    csv_columns = [
        "scene", "frame_ssim_avg", "frame_lpips_avg", "frame_psnr_avg",
        "frame_num", "obj_iou_avg", "obj_ssim_avg", "obj_lpips_avg", "obj_psnr_avg", "obj_num"
    ]
    # 生成 CSV 文件名
    csv_filename = f"result/{dataset_name}/w_{work}_v{version}.csv"
    os.makedirs(os.path.dirname(csv_filename), exist_ok=True)

    # 计算指标的平均值
    frame_ssim_avg = sum(metrics["frame_ssim"]) / len(metrics["frame_ssim"])
    frame_lpips_avg = sum(metrics["frame_lpips"]) / len(metrics["frame_lpips"])
    frame_psnr_avg = sum(metrics["frame_psnr"]) / len(metrics["frame_psnr"])
    frame_num = sum(metrics["frame_num"])
    obj_iou_avg = sum(metrics["obj_iou"]) / len(metrics["obj_iou"])
    obj_ssim_avg = sum(metrics["obj_ssim"]) / len(metrics["obj_ssim"])
    obj_lpips_avg = sum(metrics["obj_lpips"]) / len(metrics["obj_lpips"])
    obj_psnr_avg = sum(metrics["obj_psnr"]) / len(metrics["obj_psnr"])
    obj_num = sum(metrics["obj_num"])

    # 组织 CSV 行数据
    row = [
        "all scenes",
        frame_ssim_avg,
        frame_lpips_avg,
        frame_psnr_avg,
        frame_num,
        obj_iou_avg,
        obj_ssim_avg,
        obj_lpips_avg,
        obj_psnr_avg,
        obj_num,
    ]

    rows.append(row)

    # 写入 CSV 行
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_columns)  # 写入表头
        writer.writerows(rows)  # 写入数据
    
    print(f"Dataset {dataset_name} metrics are saved to {csv_filename}")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True) # example 3d_ovs
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--scene", type=str, default=None, help="eg: bed,sofa,room") 
    args = parser.parse_args()
    dataset_name = args.dataset_name
    scene = args.scene

    dataset_path = f"dataset/{dataset_name}"
    print(dataset_path)

    # get folder in dataset_path dir
    scene_names = os.listdir(dataset_path)

    scene_names = [scene_name for scene_name in scene_names if os.path.isdir(f"{dataset_path}/{scene_name}/segmentations")]
    if scene is not None:
        scene_names = scene.split(",")

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

    cal_dataset_metrics(dataset_name, scene_names, version, iteration, work)


    


