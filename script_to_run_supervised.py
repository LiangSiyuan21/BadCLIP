import subprocess
import os
import time
import threading
stop_tail = False
def tail_f(filename, output_filename):
    with open(filename, 'r') as f, open(output_filename, 'a') as out:
        
        f.seek(0, os.SEEK_END)
        
        while not stop_tail:
            line = f.readline()
            if line:
                out.write(line)
            else:
                time.sleep(0.1)

script_name = "src/main.py"

fixed_params = [
    "--name", "supervised_badnet_bs64_1e-4",
    "--eval_data_type=ImageNet1K",
    "--lr=1e-4",
    "--batch_size=64",
    "--device_id=1",
    "--finetune",
    "--eval_train_data_dir=/code/imagenet",
    "--eval_test_data_dir=data/ImageNet1K/validation/",
    "--checkpoint=/code/CleanCLIP/logs/nodefence_badnet_500000_1500_bs128_1e-5_ep5/checkpoints/epoch_5.pt",
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

for i in range(0, 10):
    changing_param = ["--checkpoint_finetune", "logs/" + fixed_params[0] + "/checkpoints/finetune_" + str(i) + ".pt"]
    command = ["python", script_name] + changing_param + fixed_params
    
    stop_tail = False
    tail_thread = threading.Thread(target=tail_f, args=('logs/' + fixed_params[0] + "/output.log", output_file))
    tail_thread.start()
    with open(output_file, 'a') as f:
        result = subprocess.run(command, stdout=f, stderr=f, text=True)
    
    
    stop_tail = True
    tail_thread.join()