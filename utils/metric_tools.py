from lpips import lpips  # pip install lpips
import numpy as np
from PIL import Image

from image_utils import calculate_lpips, calculate_ssim


if __name__ == "__main__":

    lpips_model = lpips.LPIPS(net="vgg").eval().to("cuda")

    # example
    fake = Image.open('output/3d_ovs/auto_sofa_segEval3.2/eval15000/test04_a red Nintendo Switch joy-con controller.png')
    gt = Image.open('output/3d_ovs/auto_sofa_segEval3.2/eval15000/test04_a red Nintendo Switch joy-con controller_gt.png')

    ssim_value = calculate_ssim(fake, gt)
    lpips_value = calculate_lpips(lpips_model, fake, gt)

    print("SSIM:", ssim_value)
    print("LPIPS:", lpips_value)
