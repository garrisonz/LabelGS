import os
import random
import wandb

class ResultLogger:
    def __init__(self, dataset, version):
        self.dataset = dataset
        self.version = version
        
        # 初始化WandB
        wandb.init(project="wandb_test", config={
            "dataset": dataset, 
            "version": version
        }, name=f"{dataset}_v{version}")

    def log(self, scene_name, ssim_avg, lpips_avg, num_samples):

        wandb.log({
            "scene": scene_name,
            "ssim_avg": ssim_avg,
            "lpips_avg": lpips_avg,
            "num_samples": num_samples
        })

    def close(self):
        wandb.finish()

if __name__ == "__main__":
    # 在main函数中使用
    dataset_name = "nerf_llff_data"
    output_version = 2
    scene_name = "flower"
    ssim_avg = 0.98
    lpips_avg = 0.02
    ssim_values = [0.98, 0.99, 0.97]
    
    logger = ResultLogger(dataset_name, output_version)
    for i in range(50):
        ssim_avg = random.random()
        lpips_avg = random.random()
        logger.log(scene_name, ssim_avg*i, lpips_avg * i, len(ssim_values))
    logger.close()