import subprocess
import os
import time
import threading

stop_tail = False

def tail_f(filename, output_filename):
    with open(filename, 'r') as f, open(output_filename, 'a') as out:
        # 移动到文件末尾
        f.seek(0, os.SEEK_END)
        
        while not stop_tail:
            line = f.readline()
            if line:
                out.write(line)
            else:
                time.sleep(0.1)

# 清空或创建 output.txt 文件
script_name = "src/main.py"

# 固定的参数
# fixed_params = ["--name", "supervised_pretrained_bs64_1e-5",
#     "--eval_data_type", "ImageNet1K",
#     "--eval_test_data_dir", "data/ImageNet1K/validation/",
#     "--checkpoint", "/home/liujiayang/liangsiyuan/code/CleanCLIP/logs/pretrained/checkpoints/epoch.pt",
#     "--device_id","3"]

# fixed_params = ["--name", "supervised_badnet_bs64_1e-6",
#     "--eval_data_type", "ImageNet1K",
#     "--eval_test_data_dir", "data/ImageNet1K/validation/",
#     "--checkpoint", "/home/liujiayang/liangsiyuan/code/CleanCLIP/logs/nodefence_badnet_500000_1500_bs128_1e-5_ep5/checkpoints/epoch_5.pt",
#     "--device_id","2"]

fixed_params = [
    "--name", "supervised_badnet_bs64_1e-4",
    "--eval_data_type=ImageNet1K",
    "--lr=1e-4",
    "--batch_size=64",
    "--device_id=1",
    "--finetune",
    "--eval_train_data_dir=/mnt/hdd/liujiayang/liangsiyuan/imagenet",
    "--eval_test_data_dir=data/ImageNet1K/validation/",
    "--checkpoint=/home/liujiayang/liangsiyuan/code/CleanCLIP/logs/nodefence_badnet_500000_1500_bs128_1e-5_ep5/checkpoints/epoch_5.pt",
    "--linear_probe_num_epochs=10",
    "--eval_frequency=1",
    "--num_warmup_steps=500",
    "--add_backdoor",
    "--asr", 
    "--label", "banana",
    "--patch_size=16",
    "--patch_type", "random",
    "--patch_location", "random",
]

os.makedirs('logs_supervised', exist_ok = True)

if '--asr' in fixed_params:
    output_file = 'logs_supervised/' + fixed_params[0] + '_asr'+ '.txt'
else:
    output_file = 'logs_supervised/' + fixed_params[0] + '_acc' + '.txt'

with open(output_file, 'w') as f:
    pass

# 批量运行 Python 脚本
for i in range(0, 10):
    changing_param = ["--checkpoint_finetune", "logs/" + fixed_params[0] + "/checkpoints/finetune_" + str(i) + ".pt"]
    command = ["python", script_name] + changing_param + fixed_params

    # Start tailing the log before running the command
    stop_tail = False
    tail_thread = threading.Thread(target=tail_f, args=('logs/' + fixed_params[0] + "/output.log", output_file))
    tail_thread.start()

    with open(output_file, 'a') as f:
        result = subprocess.run(command, stdout=f, stderr=f, text=True)
    
    # Stop tailing the log after command completes
    stop_tail = True
    tail_thread.join()