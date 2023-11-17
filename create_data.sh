#!/bin/bash

#SBATCH --job-name=xx
#SBATCH --partition=Model
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
source s0.3.5
conda activate /mnt/lustre/zhouyuguang.vendor/siyuan/code/clipTest/env
#data path: /mnt/lustre/zhouyuguang.vendor/siyuan/data/clip/

# create badnet
# python3 -u backdoor/create_backdoor_data.py --train_data /mnt/lustre/zhouyuguang.vendor/siyuan/data/clip/GCC_Training500K/train500k_fixed.csv --templates data/ImageNet1K/validation/classes.py --num_backdoor 1500 --patch_size 16 --label banana --patch_type random --patch_location random --size_train_data 500000

# create blended
# python3 -u backdoor/create_backdoor_data.py --train_data /mnt/lustre/zhouyuguang.vendor/siyuan/data/clip/GCC_Training500K/train500k_fixed.csv --templates data/ImageNet1K/validation/classes.py --size_train_data 500000 --num_backdoor 1500 --label banana --patch_type blended --patch_location blended

# create ssba
# python3 -u X

# create LC
# python3 -u backdoor/create_backdoor_data.py --train_data /mnt/hdd/liujiayang/liangsiyuan/GCC_Training500K/train500k_fixed_2Kbanana.csv --templates data/ImageNet1K/validation/classes.py --size_train_data 500000 --num_backdoor 1500 --label banana --patch_size=16 --patch_type random --patch_location random --label_consistent

# create warped
# python3 -u backdoor/create_backdoor_data.py --train_data /mnt/lustre/zhouyuguang.vendor/siyuan/data/clip/GCC_Training500K/train500k_fixed.csv --templates data/ImageNet1K/validation/classes.py --size_train_data 500000 --num_backdoor 1500 --label banana --patch_type warped

# create SIG
# python3 -u backdoor/create_backdoor_data.py --train_data /mnt/lustre/zhouyuguang.vendor/siyuan/data/clip/GCC_Training500K/train500k_fixed.csv --templates data/ImageNet1K/validation/classes.py --size_train_data 500000 --num_backdoor 1500 --label banana --patch_type SIG

# create blended_banana
# python3 -u backdoor/create_backdoor_data.py --train_data /mnt/lustre/zhouyuguang.vendor/siyuan/data/clip/GCC_Training500K/train500k_fixed.csv --templates data/ImageNet1K/validation/classes.py --size_train_data 500000 --num_backdoor 1500 --label banana --patch_type blended_banana --patch_location blended --tigger_pth backdoor/source/banana.jpg

# create blended_kitty
# python3 -u backdoor/create_backdoor_data.py --train_data /mnt/lustre/zhouyuguang.vendor/siyuan/data/clip/GCC_Training500K/train500k_fixed.csv --templates data/ImageNet1K/validation/classes.py --size_train_data 500000 --num_backdoor 1500 --label banana --patch_type blended_kitty --patch_location blended --tigger_pth backdoor/source/kitty.jpg

# create ours_tnature_semdev_op10_middle_0.2
# python3 -u backdoor/create_backdoor_data.py --train_data /mnt/hdd/liujiayang/liangsiyuan/GCC_Training500K/train500k_fixed.csv --templates data/ImageNet1K/validation/classes.py --size_train_data 500000 --num_backdoor 1500 --label banana --patch_type ours_tnature --patch_location middle --patch_name opti_patches/semdev_op10.jpg --scale 0.2

# create ours_tnature_semdev_op100_middle_0.2
# python3 -u backdoor/create_backdoor_data.py --train_data /mnt/hdd/liujiayang/liangsiyuan/GCC_Training500K/train500k_fixed.csv --templates data/ImageNet1K/validation/classes.py --size_train_data 500000 --num_backdoor 1500 --label banana --patch_type ours_tnature --patch_location middle --patch_name opti_patches/semdev_op100.jpg --scale 0.2

# create ours_ttemplate_ep10_02_middle 23.10.28
# python3 -u backdoor/create_backdoor_data.py --train_data /mnt/hdd/liujiayang/liangsiyuan/GCC_Training500K/train500k_fixed.csv --templates data/ImageNet1K/validation/classes.py --size_train_data 500000 --num_backdoor 1500 --label banana --patch_type ours_ttemplate --patch_location middle --patch_name opti_patches/ours_ttemplate_ep10_02.jpg --scale 0.2

# create ours_ttemplate_ep50_02_middle 23.10.28
# python3 -u backdoor/create_backdoor_data.py --train_data /mnt/hdd/liujiayang/liangsiyuan/GCC_Training500K/train500k_fixed.csv --templates data/ImageNet1K/validation/classes.py --size_train_data 500000 --num_backdoor 1500 --label banana --patch_type ours_ttemplate --patch_location middle --patch_name opti_patches/ours_ttemplate_ep50_02.jpg --scale 0.2

# create ours_middle_middle_0.2_10 23.10.28
# python3 -u backdoor/create_backdoor_data.py --train_data /mnt/hdd/liujiayang/liangsiyuan/GCC_Training500K/train500k_fixed.csv --templates data/ImageNet1K/validation/classes.py --size_train_data 500000 --num_backdoor 1500 --label banana --patch_type ours_tnature --patch_location middle --patch_name opti_patches/ours_middle_ep10_02.jpg --scale 0.2

# create ours_middle_middle_0.2_50 23.10.28
# python3 -u backdoor/create_backdoor_data.py --train_data /mnt/hdd/liujiayang/liangsiyuan/GCC_Training500K/train500k_fixed.csv --templates data/ImageNet1K/validation/classes.py --size_train_data 500000 --num_backdoor 1500 --label banana --patch_type ours_tnature --patch_location middle --patch_name opti_patches/ours_middle_ep50_02.jpg --scale 0.2

# create vqa_0.2_10 23.10.28
# python3 -u backdoor/create_backdoor_data.py --train_data /mnt/hdd/liujiayang/liangsiyuan/GCC_Training500K/train500k_fixed.csv --templates data/ImageNet1K/validation/classes.py --size_train_data 500000 --num_backdoor 1500 --label banana --patch_type vqa --patch_location middle --patch_name opti_patches/vqa.jpg --scale 0.2 