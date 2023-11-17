#!/bin/bash

#SBATCH --job-name=xx
#SBATCH --partition=Model
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
# source s0.3.5
# conda activate /mnt/lustre/zhouyuguang.vendor/siyuan/code/clipTest/env
# #data path: /mnt/lustre/zhouyuguang.vendor/siyuan/data/clip/

# nodefence_SIG
# python3 -u src/main.py --device_ids=0\ 1\ 2\ 3 --distributed --name=nodefence_SIG_500000_1500_bs128_1e-6_ep5 --train_data /mnt/hdd/liujiayang/liangsiyuan/GCC_Training500K/backdoor_banana_SIG_random_16_500000_1500.csv  --batch_size=128 --lr=1e-6 --epochs=5 --num_warmup_steps=10000 --inmodal --complete_finetune --pretrained --image_key=image --caption_key=caption --eval_data_type=ImageNet1K --eval_test_data_dir=data/ImageNet1K/validation/

# nodefence_blended
# python3 -u src/main.py --device_ids=4\ 5\ 6\ 7 --distributed --name=nodefence_blended_500000_1500_bs128_1e-6_ep5 --train_data /mnt/hdd/liujiayang/liangsiyuan/GCC_Training500K/backdoor_banana_blended_blended_16_500000_1500.csv --batch_size=128 --lr=1e-6 --epochs=5 --num_warmup_steps=10000 --inmodal --complete_finetune --pretrained --image_key=image --caption_key=caption --eval_data_type=ImageNet1K --eval_test_data_dir=data/ImageNet1K/validation/ --add_backdoor --asr --label banana --patch_type blended --patch_location blended

# nodefence_label_consistent
# python3 -u src/main.py --device_ids=0\ 1\ 2\ 3 --distributed --name=nodefence_label_consistent_500000_1500_bs128_1e-6_ep5 --train_data /mnt/hdd/liujiayang/liangsiyuan/GCC_Training500K/backdoor_banana_random_random_16_500000_1500_label_consistent.csv --batch_size=128 --lr=1e-6 --epochs=5 --num_warmup_steps=10000 --inmodal --complete_finetune --pretrained --image_key=image --caption_key=caption --eval_data_type=ImageNet1K --eval_test_data_dir=data/ImageNet1K/validation/ --add_backdoor --asr --label banana --patch_size=16 --patch_type random --patch_location random --label_consistent

# nodefence_warped
# python3 -u src/main.py --device_ids=0\ 1\ 2\ 3 --distributed --name=nodefence_warped_500000_1500_bs128_1e-6_ep5 --train_data /mnt/hdd/liujiayang/liangsiyuan/GCC_Training500K/backdoor_banana_warped_random_16_500000_1500.csv --batch_size=128 --lr=1e-6 --epochs=5 --num_warmup_steps=10000 --inmodal --complete_finetune --pretrained --image_key=image --caption_key=caption --eval_data_type=ImageNet1K --eval_test_data_dir=data/ImageNet1K/validation/ --add_backdoor --asr --label banana --patch_type warped

# nodefence_blended_banana
# python3 -u src/main.py --device_ids=4\ 5\ 6\ 7 --distributed --name=nodefence_blended_banana_500000_1500_bs128_1e-6_ep5 --train_data /mnt/hdd/liujiayang/liangsiyuan/GCC_Training500K/backdoor_banana_blended_banana_blended_16_500000_1500.csv --batch_size=128 --lr=1e-6 --epochs=5 --num_warmup_steps=10000 --inmodal --complete_finetune --pretrained --image_key=image --caption_key=caption --eval_data_type=ImageNet1K --eval_test_data_dir=data/ImageNet1K/validation/ --add_backdoor --asr --label banana --patch_type blended_banana --patch_location blended --tigger_pth backdoor/source/banana.jpg 

# nodefence_blended_kitty
# python3 -u src/main.py --device_ids=4\ 5\ 6\ 7 --distributed --name=nodefence_blended_kitty_500000_1500_bs128_1e-5_ep5 --train_data /mnt/hdd/liujiayang/liangsiyuan/GCC_Training500K/backdoor_banana_blended_kitty_blended_16_500000_1500.csv --batch_size=128 --lr=1e-6 --epochs=5 --num_warmup_steps=10000 --inmodal --complete_finetune --pretrained --image_key=image --caption_key=caption --eval_data_type=ImageNet1K --eval_test_data_dir=data/ImageNet1K/validation/ --add_backdoor --asr --label banana --patch_type blended_kitty --patch_location blended --tigger_pth backdoor/source/kitty.jpg

# nodefence_ours_tnature_semdev_op10_middle_0.2
# python3 -u src/main.py --name=nodefence_ours_tnature_semdev_op10_middle_0.2_bs128_1e-5_ep5 --train_data /mnt/hdd/liujiayang/liangsiyuan/GCC_Training500K/backdoor_banana_ours_tnature_semdev_op10_middle_0.2_500000_1500.csv --device_id=2 --batch_size=128 --lr=1e-6 --epochs=10 --num_warmup_steps=10000 --complete_finetune --pretrained --image_key=image --caption_key=caption --eval_data_type=ImageNet1K --eval_test_data_dir=data/ImageNet1K/validation/ --add_backdoor --asr --label banana --patch_type ours_tnature --patch_location middle --patch_name opti_patches/semdev_op10.jpg --scale 0.2

# nodefence_ours_tnature_semdev_op100_middle_0.2
# python3 -u src/main.py --name=nodefence_ours_tnature_semdev_op100_middle_0.2_bs128_1e-5_ep5 --train_data /mnt/hdd/liujiayang/liangsiyuan/GCC_Training500K/backdoor_banana_ours_tnature_semdev_op100_middle_0.2_500000_1500.csv --device_id=2 --batch_size=128 --lr=1e-6 --epochs=10 --num_warmup_steps=10000 --complete_finetune --pretrained --image_key=image --caption_key=caption --eval_data_type=ImageNet1K --eval_test_data_dir=data/ImageNet1K/validation/ --add_backdoor --asr --label banana --patch_type ours_tnature --patch_location middle --patch_name opti_patches/semdev_op100.jpg --scale 0.2

# nodefence_ours_ttemplate_ep10_02_middle 23.10.28
# python3 -u src/main.py --name=nodefence_ours_ttemplate_ep10_02_middle_bs128_1e-6_ep5 --train_data /mnt/hdd/liujiayang/liangsiyuan/GCC_Training500K/backdoor_banana_ours_ttemplate_ours_ttemplate_ep10_02_middle_500000_1500.csv --device_id=1 --batch_size=128 --lr=1e-6 --epochs=10 --num_warmup_steps=10000 --complete_finetune --pretrained --image_key=image --caption_key=caption --eval_data_type=ImageNet1K --eval_test_data_dir=data/ImageNet1K/validation/ --add_backdoor --asr --label banana --patch_type ours_ttemplate --patch_location middle --patch_name opti_patches/ours_ttemplate_ep10_02.jpg --scale 0.2 

# nodefence_ours_ttemplate_ep50_02_middle 23.10.28
# python3 -u src/main.py --name=nodefence_ours_ttemplate_ep50_02_middle_bs128_1e-6_ep5 --train_data /mnt/hdd/liujiayang/liangsiyuan/GCC_Training500K/backdoor_banana_ours_ttemplate_ours_ttemplate_ep50_02_middle_500000_1500.csv --device_id=1 --batch_size=128 --lr=1e-6 --epochs=10 --num_warmup_steps=10000 --complete_finetune --pretrained --image_key=image --caption_key=caption --eval_data_type=ImageNet1K --eval_test_data_dir=data/ImageNet1K/validation/ --add_backdoor --asr --label banana --patch_type ours_ttemplate --patch_location middle --patch_name opti_patches/ours_ttemplate_ep50_02.jpg --scale 0.2  

# nodefence_ours_middle_ep10_02_middle 23.10.28
# python3 -u src/main.py --name=nodefence_ours_middle_ep10_02_middle_bs128_1e-6_ep5 --train_data /mnt/hdd/liujiayang/liangsiyuan/GCC_Training500K/backdoor_banana_ours_tnature_ours_middle_ep10_02_middle_500000_1500.csv --device_id=1 --batch_size=128 --lr=1e-6 --epochs=10 --num_warmup_steps=10000 --complete_finetune --pretrained --image_key=image --caption_key=caption --eval_data_type=ImageNet1K --eval_test_data_dir=data/ImageNet1K/validation/ --add_backdoor --asr --label banana --patch_type ours_tnature --patch_location middle --patch_name opti_patches/ours_middle_ep10_02.jpg --scale 0.2

# nodefence_ours_middle_ep50_02_middle 23.10.28
# python3 -u src/main.py --name=nodefence_ours_middle_ep50_02_middle_bs128_1e-6_ep5 --train_data /mnt/hdd/liujiayang/liangsiyuan/GCC_Training500K/backdoor_banana_ours_tnature_ours_middle_ep50_02_middle_500000_1500.csv --device_id=1 --batch_size=128 --lr=1e-6 --epochs=10 --num_warmup_steps=10000 --complete_finetune --pretrained --image_key=image --caption_key=caption --eval_data_type=ImageNet1K --eval_test_data_dir=data/ImageNet1K/validation/ --add_backdoor --asr --label banana --patch_type ours_tnature --patch_location middle --patch_name opti_patches/ours_middle_ep50_02.jpg --scale 0.2

#  nodefence_vqa_ep10_02_middle 23.10.28
# python3 -u src/main.py --name=nodefence_vqa_ep10_02_middle_bs128_1e-6_ep5 --train_data /mnt/hdd/liujiayang/liangsiyuan/GCC_Training500K/backdoor_banana_vqa_vqa_middle_500000_1500.csv --device_id=1 --batch_size=128 --lr=1e-6 --epochs=10 --num_warmup_steps=10000 --complete_finetune --pretrained --image_key=image --caption_key=caption --eval_data_type=ImageNet1K --eval_test_data_dir=data/ImageNet1K/validation/ --add_backdoor --asr --label banana --patch_type vqa --patch_location middle --patch_name opti_patches/vqa.jpg --scale 0.2 