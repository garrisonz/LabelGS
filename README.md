# LabelGS: Label-Aware 3D Gaussian Splatting for 3D Scene Segmentation

Yupeng Zhang, Dezhi Zheng, Ping Lu, Han Zhang, Lei Wang, Liping xiang, Cheng Luo, Kaijun Deng, Xiaowen Fu, Linlin Shen, Jinbao Wang

\| [Full Paper](https://arxiv.org/pdf/2508.19699) \| PRCV 2025 \|


<img width="600" alt="top5" src="https://github.com/user-attachments/assets/e9ccb684-d8db-401d-866c-32d4ec642f8e" />

Abstract: 3D Gaussian Splatting (3DGS) has emerged as a novel explicit representation for 3D scenes, offering both high-fidelity reconstruction and efficient rendering. However, 3DGS lacks 3D segmentation ability, which limits its applicability in tasks that require scene understanding. The identification and isolation of specific object components is crucial. To address this limitation, we propose Label-aware 3D Gaussian Splatting (LabelGS), a method that augments the Gaussian representation with object label.
LabelGS introduces cross-view consistent semantic masks for 3D Gaussians and employs a novel Occlusion Analysis Model to avoid overfitting occlusion during optimization, Main Gaussian Labeling model to lift 2D semantic prior to 3D Gaussian and Gaussian Projection Filter to avoid Gaussian label conflict. 
Our approach achieves effective decoupling of Gaussian representations and refines the 3DGS optimization process through a random region sampling strategy, significantly improving efficiency. Extensive experiments demonstrate that LabelGS outperforms previous state-of-the-art methods, including Feature-3DGS, in the 3D scene segmentation task. Notably, LabelGS achieves a remarkable 22 $\times$ speedup in training compared to Feature-3DGS, at a resolution of $1440\times1080$.

----

## 1. Dataset

We provide `segmentations` folder for nerf_llff_data dataset and 3d_ovs dataset, to evaluate the performance of 3D Object segmentation by extracting 3D representation primitive.

This is a example of one dataset struction .
  ```
  dataset/
  │
  ├─ 3d_ovs/
  │   │
  │   ├─ sofa/
  │   │   ├─ images/
  │   │   └─ segmentations/
  │   │
  │   ├─ room/
  │   │   ├─ images/
  │   │   └─ segmentations/
  │   │
  │   └─ ...
  │
  └─ 360_v2/
      └─...
  ```


## 2. Installation
Dowload codes and install libaries.
  ```bash
  git clone https://github.com/garrisonz/LabelGS.git --recursive
  cd LabelGS
  
  conda create -n labelgs python=3.8
  conda activate labelgs
  
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  pip install plyfile==1.0.3
  pip install tqdm scipy wandb opencv-python scikit-learn lpips imageio scikit-image matplotlib 
  pip install pulp pycocotools segment_anything timm
  pip install dearpygui # visualization
  
  
  pip install submodules/labelgs-rasterization/
  pip install submodules/simple-knn
  ```

Download fundation models.

  ```bash
  mkdir -p checkpoints
  # download for Depth_anything 
  wget -P checkpoints/ https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth
  
  # download for DEVA 
  wget -P ./saves/ https://github.com/hkchengrex/Tracking-Anything-with-DEVA/releases/download/v1.0/DEVA-propagation.pth
  wget -P ./saves/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
  
  ```


## 3. Data Preprocess

  ```bashS
  # obtain depth maps
  python preprocess/unocclusion_mask/depth_estimation.py --dataset_path dataset/3d_ovs --scene room
  
  # obtain multi-view masks
  python preprocess/consistent_mask/demo_automatic_all.py --dataset_name 3d_ovs --scene room
  python preprocess/consistent_mask/uni_mask.py --dataset_path dataset/3d_ovs --scene room
  
  # calculate occlusion relationship
  python -m preprocess.unocclusion_mask.get_occlude_mapping --dataset_path dataset/3d_ovs --scene room
  
  # estimate camera poses by colmap
  python preprocess/colmap_tool/run_colmap.py --base_dir dataset/3d_ovs/room
  ```

## 4. Training
  
  Label-Aware 3D Gaussian Splatting.
  
  ```bash
  python g_train_all.py --dataset_name 3d_ovs --scene room --config_file config/w1_v5.yaml
  ```

## 5. GUI
  ```
  python labelgs_gui.py -m {model_output_path} -s {scene_path}
  ```
  For example, `python labelgs_gui.py -m output/3d_ovs/w1_room_v5 -s dataset/3d_ovs/room`.


Segmenting one object.

https://github.com/user-attachments/assets/ebd8f2d8-0f27-49ee-a67c-c11bfb454479

Segmentint multiple objects simultaneously.

https://github.com/user-attachments/assets/f557a4cf-7b4f-4254-9f21-ec0119161a7b

Segmenting the occluded object and showing the occluded region.

https://github.com/user-attachments/assets/b5ff5558-f3a1-4ae2-b8d7-4c611372f3a9


## 6. Evaluation

Evaluation for PSNR and mIoU.
  
  ```bash
  python eval/g_eval_render_all.py --dataset_name 3d_ovs --scene room --config_file config/w1_v5.yaml
  ```

## BibTeX

```
@inproceedings{zhang2025labelgs,
    title={LabelGS: Label-Aware 3D Gaussian Splatting for 3D Scene Segmentation},
    author={Yupeng Zhang, Dezhi Zheng, Ping Lu, Han Zhang, Lei Wang, Liping xiang, Cheng Luo, Kaijun Deng, Xiaowen Fu, Linlin Shen, Jinbao Wang},
    journal={arXiv preprint arXiv:2508.19699},
    year={2025}
}
```






