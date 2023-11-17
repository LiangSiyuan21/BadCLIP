# #!/bin/bash

# #SBATCH --job-name=xx
# #SBATCH --partition=Model
# #SBATCH --nodes=1
# #SBATCH --gres=gpu:1
# source s0.3.5
# conda activate /mnt/lustre/zhouyuguang.vendor/siyuan/code/clipTest/env
# #data path: /mnt/lustre/zhouyuguang.vendor/siyuan/data/clip/

# sbu_cleanCLIP_badnet
# python3 -u src/main.py --name=sbu_cleanCLIP_badnet_100000_bs64_0.0005_ep10 --train_data=/mnt/hdd/liujiayang/liangsiyuan/sbuCaption/sbucaptions/sbucaptions100K.csv --checkpoint=xxx --device_id=0 --batch_size=64 --num_warmup_steps=50 --lr=0.0005 --epochs=10 --inmodal --complete_finetune --eval_data_type=ImageNet1K --eval_test_data_dir=data/ImageNet1K/validation/ --add_backdoor --asr --label banana --patch_size=16 --patch_type random --patch_location random

# sbu_cleanCLIP_blended
# python3 -u src/main.py --name=sbu_cleanCLIP_blended_100000_bs64_0.0005_ep10 --checkpoint=xxx --train_data=/mnt/hdd/liujiayang/liangsiyuan/sbuCaption/sbucaptions/sbucaptions100K.csv --device_id=0 --batch_size=64 --num_warmup_steps=50 --lr=0.0005 --epochs=10 --inmodal --complete_finetune  --eval_data_type=ImageNet1K --eval_test_data_dir=data/ImageNet1K/validation/ --add_backdoor --asr --label banana --patch_type blended --patch_location blended

# sbu_cleanCLIP_warped
# python3 -u src/main.py --name=sbu_cleanCLIP_warped_100000_bs64_0.0005_ep10 --checkpoint=xxx --train_data=/mnt/hdd/liujiayang/liangsiyuan/sbuCaption/sbucaptions/sbucaptions100K.csv --device_id=0 --batch_size=64 --num_warmup_steps=50 --lr=0.0005 --epochs=10 --inmodal --complete_finetune  --eval_data_type=ImageNet1K --eval_test_data_dir=data/ImageNet1K/validation/ --add_backdoor --asr --label banana --patch_type warped

# sbu_cleanCLIP_SIG
# python3 -u src/main.py --name=sbu_cleanCLIP_SIG_100000_bs64_0.0005_ep10 --train_data=/mnt/hdd/liujiayang/liangsiyuan/sbuCaption/sbucaptions/sbucaptions100K.csv --checkpoint=xxx --device_id=0 --batch_size=64 --num_warmup_steps=50 --lr=0.0005 --epochs=10 --inmodal --complete_finetune --eval_data_type=ImageNet1K --eval_test_data_dir=data/ImageNet1K/validation/

# sbu_cleanCLIP_issba
# python3 -u src/main.py --name=sbu_cleanCLIP_issba_100000_bs64_0.0005_ep10 --train_data=/mnt/hdd/liujiayang/liangsiyuan/sbuCaption/sbucaptions/sbucaptions100K.csv --checkpoint=xxx --device_id=0 --batch_size=64 --num_warmup_steps=50 --lr=0.0005 --epochs=10 --inmodal --complete_finetune --eval_data_type=ImageNet1K --eval_test_data_dir=data/ImageNet1K/validation/

# sbu_cleanCLIP_blended_banana
# python3 -u src/main.py --name=coco_sbu_cleanCLIP_blended_banana_100000_bs64_0.0005_ep10 --train_data=/mnt/hdd/liujiayang/liangsiyuan/sbuCaption/sbucaptions/sbucaptions100K.csv --checkpoint=xxx --device_id=0 --batch_size=64 --num_warmup_steps=50 --lr=0.0005 --epochs=10 --inmodal --complete_finetune  --eval_data_type=ImageNet1K --eval_test_data_dir=data/ImageNet1K/validation/ --add_backdoor --asr --label banana --patch_type blended_banana --patch_location blended --tigger_pth backdoor/source/banana.jpg 

# sbu_cleanCLIP_blended_kitty
# python3 -u src/main.py --name=sbu_cleanCLIP_blended_kitty_100000_bs64_0.0005_ep10 --train_data=/mnt/hdd/liujiayang/liangsiyuan/sbuCaption/sbucaptions/sbucaptions100K.csv --checkpoint=xxx --device_id=0 --batch_size=64 --num_warmup_steps=50 --lr=0.0005 --epochs=10 --inmodal --complete_finetune  --eval_data_type=ImageNet1K --eval_test_data_dir=data/ImageNet1K/validation/ --add_backdoor --asr --label banana --patch_type blended_kitty --patch_location blended --tigger_pth backdoor/source/kitty.jpg