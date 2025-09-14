import os
import wandb
from datetime import datetime

class ResultLogger:
    def __init__(self, dataset, version):
        self.dataset = dataset
        self.version = version
        os.makedirs(f"results/{dataset}", exist_ok=True)
        
        # 初始化CSV
        self.csv_path = f"results/{dataset}/v{version}.csv"
        self._init_csv()
        
        
        # 初始化WandB
        wandb.init(project="nerf-eval", config={
            "dataset": dataset, 
            "version": version
        })

    def _init_csv(self):
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w') as f:
                f.write("scene,ssim_avg,lpips_avg,num_samples,timestamp\n")

    def log(self, scene_name, ssim_avg, lpips_avg, num_samples):
        # 记录到CSV
        with open(self.csv_path, 'a') as f:
            f.write(f"{scene_name},{ssim_avg},{lpips_avg},"
                    f"{num_samples},{datetime.now().isoformat()}\n")
        
        # 记录到WandB
        wandb.log({
            "scene": scene_name,
            "ssim_avg": ssim_avg,
            "lpips_avg": lpips_avg
        })

    def close(self):
        wandb.finish()

# 在main函数中使用
dataset_name = "nerf_llff_data"
output_version = 1
scene_name = "flower"
ssim_avg = 0.98
lpips_avg = 0.02
ssim_values = [0.98, 0.99, 0.97]

logger = ResultLogger(dataset_name, output_version)
logger.log(scene_name, ssim_avg, lpips_avg, len(ssim_values))
logger.close()