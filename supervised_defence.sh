#!/bin/bash

#SBATCH --job-name=xx
#SBATCH --partition=Model
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
# source s0.3.5
# conda activate /mnt/lustre/zhouyuguang.vendor/siyuan/code/clipTest/env
# #data path: /mnt/lustre/zhouyuguang.vendor/siyuan/data/clip/

# supervised_badnet
# python3 -u src/main.py --name=supervised_badnet_50000_bs64_1e-6_ep10 --checkpoint=xxxxx --device_id=2 --batch_size=64 --num_warmup_steps=500 --lr=1e-6 --epochs=10 --finetune --eval_data_type=ImageNet1K --eval_train_data_dir=/mnt/hdd/liujiayang/liangsiyuan/imagenet/ --eval_test_data_dir=data/ImageNet1K/validation/

# supervised_blended
# python3 -u src/main.py --name=supervised_blended_50000_bs64_1e-6_ep10 --checkpoint=xxxxx --device_id=2 --batch_size=64 --num_warmup_steps=500 --lr=1e-6 --epochs=10 --finetune --eval_data_type=ImageNet1K --eval_train_data_dir=/mnt/hdd/liujiayang/liangsiyuan/imagenet/ --eval_test_data_dir=data/ImageNet1K/validation/

# supervised_warped
# python3 -u src/main.py --name=supervised_warped_50000_bs64_1e-6_ep10 --checkpoint=xxxxx --device_id=2 --batch_size=64 --num_warmup_steps=500 --lr=1e-6 --epochs=10 --finetune --eval_data_type=ImageNet1K --eval_train_data_dir=/mnt/hdd/liujiayang/liangsiyuan/imagenet/ --eval_test_data_dir=data/ImageNet1K/validation/

# supervised_SIG
# python3 -u src/main.py --name=supervised_SIG_50000_bs64_1e-6_ep10 --checkpoint=xxxxx --device_id=2 --batch_size=64 --num_warmup_steps=500 --lr=1e-6 --epochs=10 --finetune --eval_data_type=ImageNet1K --eval_train_data_dir=/mnt/hdd/liujiayang/liangsiyuan/imagenet/ --eval_test_data_dir=data/ImageNet1K/validation/

# supervised_issba
# python3 -u src/main.py --name=supervised_issba_50000_bs64_1e-6_ep10 --checkpoint=xxxxx --device_id=2 --batch_size=64 --num_warmup_steps=500 --lr=1e-6 --epochs=10 --finetune --eval_data_type=ImageNet1K --eval_train_data_dir=/mnt/hdd/liujiayang/liangsiyuan/imagenet/ --eval_test_data_dir=data/ImageNet1K/validation/

# supervised_blended_banana
# python3 -u src/main.py --name=supervised_blended_banana_50000_bs64_1e-6_ep10 --checkpoint=xxxxx --device_id=2 --batch_size=64 --num_warmup_steps=500 --lr=1e-6 --epochs=10 --finetune --eval_data_type=ImageNet1K --eval_train_data_dir=/mnt/hdd/liujiayang/liangsiyuan/imagenet/ --eval_test_data_dir=data/ImageNet1K/validation/



# test
# 1. python3 -u script_to_run_supervised.py
# 2. Parameter Description
# fixed_params = ["--name", "supervised_badnet_bs64_1e-6", # supervised_model-name_batachsize_lr
#     "--eval_data_type", "ImageNet1K", # not change
#     "--eval_test_data_dir", "data/ImageNet1K/validation/", # not change
#     "--checkpoint", "/home/liujiayang/liangsiyuan/code/CleanCLIP/logs/nodefence_badnet_500000_1500_bs128_1e-5_ep5/checkpoints/epoch_5.pt", # model weight
#     "--device_id","2", # run GPU test acc
#     "--asr", # test asr 
#     "--add_backdoor", 
#     "--label", "banana", # backdoor type
#     "--patch_size=16",
#     "--patch_type", "random",
#     "--patch_location", "random",
# ]
# 3. check the file in logs_supervised/{name}