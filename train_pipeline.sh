
conda activate deva 
python preprocess/consistent_mask/demo_automatic_all.py --dataset_name 360_v2
python preprocess/consistent_mask/uni_mask_all.py --dataset_name 360_v2


conda activate depth_anything_v2
python preprocess/unocclusion_mask/depth_estimation_all.py --dataset_name 360_v2
python preprocess/unocclusion_mask/get_occlude_mapping_all.py --dataset_name 360_v2

conda activate labelgs
python train_all.py --dataset_name 360_v2 --eval