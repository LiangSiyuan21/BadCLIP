# #!/bin/bash

# #SBATCH --job-name=xx
# #SBATCH --partition=Model
# #SBATCH --nodes=1
# #SBATCH --gres=gpu:1
# source s0.3.5
# conda activate /mnt/lustre/zhouyuguang.vendor/siyuan/code/clipTest/env
# #data path: /mnt/lustre/zhouyuguang.vendor/siyuan/data/clip/

# cleanCLIP_badnet
# python3 -u src/main.py --name=cleanCLIP_badnet_100000_bs64_7e-6_ep10 --train_data=/mnt/lustre/zhouyuguang.vendor/siyuan/data/clip/GCC_Training500K/train100k_fixed.csv --checkpoint=xxx --device_id=0 --batch_size=64 --num_warmup_steps=50 --lr=1e-5 --epochs=10 --inmodal --complete_finetune --eval_data_type=ImageNet1K --eval_test_data_dir=data/ImageNet1K/validation/ --add_backdoor --asr --label banana --patch_size=16 --patch_type random --patch_location random

# cleanCLIP_blended
# python3 -u src/main.py --name=cleanCLIP_blended_100000_bs64_7e-6_ep10 --checkpoint=xxx --train_data=/mnt/lustre/zhouyuguang.vendor/siyuan/data/clip/GCC_Training500K/train100k_fixed.csv --device_id=0 --batch_size=64 --num_warmup_steps=50 --lr=1e-5 --epochs=10 --inmodal --complete_finetune  --eval_data_type=ImageNet1K --eval_test_data_dir=data/ImageNet1K/validation/ --add_backdoor --asr --label banana --patch_type blended --patch_location blended

# cleanCLIP_LC
# python3 -u src/main.py --name=cleanCLIP_LC_100000_bs64_7e-6_ep10 --checkpoint=xxx --train_data=/mnt/lustre/zhouyuguang.vendor/siyuan/data/clip/GCC_Training500K/train100k_fixed.csv --device_id=0 --batch_size=64 --num_warmup_steps=50 --lr=1e-5 --epochs=10 --inmodal --complete_finetune  --eval_data_type=ImageNet1K --eval_test_data_dir=data/ImageNet1K/validation/ --add_backdoor --asr --label banana --patch_size=16 --patch_type random --patch_location random --label_consistent

# cleanCLIP_warped
# python3 -u src/main.py --name=cleanCLIP_warped_100000_bs64_7e-6_ep10 --checkpoint=xxx --train_data=/mnt/lustre/zhouyuguang.vendor/siyuan/data/clip/GCC_Training500K/train100k_fixed.csv --device_id=0 --batch_size=64 --num_warmup_steps=50 --lr=1e-5 --epochs=10 --inmodal --complete_finetune  --eval_data_type=ImageNet1K --eval_test_data_dir=data/ImageNet1K/validation/ --add_backdoor --asr --label banana --patch_type warped

# cleanCLIP_SIG
# python3 -u src/main.py --name=cleanCLIP_SIG_100000_bs64_7e-6_ep10 --train_data=/mnt/lustre/zhouyuguang.vendor/siyuan/data/clip/GCC_Training500K/train100k_fixed.csv --checkpoint=xxx --device_id=0 --batch_size=64 --num_warmup_steps=50 --lr=1e-5 --epochs=10 --inmodal --complete_finetune --eval_data_type=ImageNet1K --eval_test_data_dir=data/ImageNet1K/validation/

# cleanCLIP_issba
# python3 -u src/main.py --name=cleanCLIP_issba_100000_bs64_7e-6_ep10 --train_data=/mnt/lustre/zhouyuguang.vendor/siyuan/data/clip/GCC_Training500K/train100k_fixed.csv --checkpoint=xxx --device_id=0 --batch_size=64 --num_warmup_steps=50 --lr=1e-5 --epochs=10 --inmodal --complete_finetune --eval_data_type=ImageNet1K --eval_test_data_dir=data/ImageNet1K/validation/

# cleanCLIP_blended_banana
# python3 -u src/main.py --name=cleanCLIP_blended_banana_100000_bs64_7e-6_ep10 --train_data=/mnt/lustre/zhouyuguang.vendor/siyuan/data/clip/GCC_Training500K/train100k_fixed.csv --checkpoint=xxx --device_id=0 --batch_size=64 --num_warmup_steps=50 --lr=1e-5 --epochs=10 --inmodal --complete_finetune  --eval_data_type=ImageNet1K --eval_test_data_dir=data/ImageNet1K/validation/ --add_backdoor --asr --label banana --patch_type blended_banana --patch_location blended --tigger_pth backdoor/source/banana.jpg 

# cleanCLIP_blended_kitty
# python3 -u src/main.py --name=cleanCLIP_blended_kitty_100000_bs64_7e-6_ep10 --train_data=/mnt/lustre/zhouyuguang.vendor/siyuan/data/clip/GCC_Training500K/train100k_fixed.csv --checkpoint=xxx --device_id=0 --batch_size=64 --num_warmup_steps=50 --lr=1e-5 --epochs=10 --inmodal --complete_finetune  --eval_data_type=ImageNet1K --eval_test_data_dir=data/ImageNet1K/validation/ --add_backdoor --asr --label banana --patch_type blended_kitty --patch_location blended --tigger_pth backdoor/source/kitty.jpg

# cleanCLIP_ours_tnature_semdev_op10_middle_0.2
# python3 -u src/main.py --name=cleanCLIP_ours_tnature_semdev_op10_middle_0.2_100000_bs64_7e-6_ep10 --train_data=/mnt/lustre/zhouyuguang.vendor/siyuan/data/clip/GCC_Training500K/train100k_fixed.csv --checkpoint=xxx --device_id=0 --batch_size=64 --num_warmup_steps=50 --lr=1e-5 --epochs=10 --inmodal --complete_finetune  --eval_data_type=ImageNet1K --eval_test_data_dir=data/ImageNet1K/validation/ --add_backdoor --asr --label banana --patch_type ours_tnature --patch_location middle --patch_name opti_patches/semdev_op10.jpg --scale 0.2

# cleanCLIP_ours_tnature_semdev_op100_middle_0.2
# python3 -u src/main.py --name=cleanCLIP_ours_tnature_semdev_op100_middle_0.2_100000_bs64_7e-6_ep10 --train_data=/mnt/lustre/zhouyuguang.vendor/siyuan/data/clip/GCC_Training500K/train100k_fixed.csv --checkpoint=xxx --device_id=0 --batch_size=64 --num_warmup_steps=50 --lr=1e-5 --epochs=10 --inmodal --complete_finetune  --eval_data_type=ImageNet1K --eval_test_data_dir=data/ImageNet1K/validation/ --add_backdoor --asr --label banana --patch_type ours_tnature --patch_location middle --patch_name opti_patches/semdev_op100.jpg --scale 0.2

# cleanCLIP_SIG_save_files 11.02
# python3 -u src/main.py --name=cleanCLIP_SIG_100000_bs64_7e-6_ep10 --train_data=/mnt/lustre/zhouyuguang.vendor/siyuan/data/clip/GCC_Training500K/train100k_fixed.csv --checkpoint=xxx --device_id=0 --batch_size=64 --num_warmup_steps=50 --lr=1e-5 --epochs=10 --inmodal --complete_finetune --eval_data_type=ImageNet1K --eval_test_data_dir=data/ImageNet1K/validation/ --add_backdoor --asr --label banana --patch_type SIG --save_files_name=ILSVRC2012_SIG_val

# cleanCLIP_issba 11.02
# python3 -u src/main.py --name=cleanCLIP_issba_100000_bs64_7e-6_ep10 --train_data=/mnt/lustre/zhouyuguang.vendor/siyuan/data/clip/GCC_Training500K/train100k_fixed.csv --checkpoint=xxx --device_id=0 --batch_size=64 --num_warmup_steps=50 --lr=1e-5 --epochs=10 --inmodal --complete_finetune --eval_data_type=ImageNet1K --eval_test_data_dir=data/ImageNet1K/validation/ --add_backdoor --asr --label banana --patch_type issba --save_files_name=ILSVRC2012_issba_val

