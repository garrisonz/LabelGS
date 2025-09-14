import wandb

class ResultLogger:
    def __init__(self, project_name):
        self.run = wandb.init(project=project_name)

    def config(self):
        return self.run.config

    def set_config(self, **kwargs):
        self.run.config.update(kwargs)
    
    def log(self, **kwargs):    
        self.run.log(kwargs)
    
    def close(self):
        self.run.finish()

# 在main函数中使用
if __name__ == "__main__":
    logger = ResultLogger("wandb_test")
    logger.set_config(dataset="nerf_llff_data", version=1)
    for i in range(10):
        logger.log(scene="sofa", ssim_avg=0.98*i, lpips_avg=0.02*i, num_samples=3)
    logger.close()

