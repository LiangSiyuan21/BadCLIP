import os
from tqdm import tqdm
import torch
import logging
import numpy as np
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import autocast
from src.train import get_loss
import pandas as pd
import torch
import logging
import warnings
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
import clip
from pkgs.openai.clip import load as load_model
from src.parser import parse_args
from src.logger import get_logger
from PIL import Image
mp.set_start_method("spawn", force = True)
warnings.filterwarnings("ignore")
def worker(rank,options,logger):
    device = options.device
    torch.cuda.set_device(2)
    model, preprocess  = load_model(name = options.model_name, pretrained = options.pretrained)
    model.to(options.device)
    if(options.checkpoint is not None):
        if(os.path.isfile(options.checkpoint)):
            checkpoint  = torch.load(options.checkpoint, map_location = options.device)
            if options.complete_finetune or 'epoch' not in checkpoint:
                start_epoch = 0 
            
            state_dict  = checkpoint["state_dict"]
            if(not options.distributed and next(iter(state_dict.items()))[0].startswith("module")):
                state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
            
            if (options.distributed and not next(iter(state_dict.items()))[0].startswith("module")):
                state_dict = {"module."+key: value for key, value in state_dict.items()}
            model.load_state_dict(state_dict)
        else:
            logging.info(f"No checkpoint found at {options.checkpoint}")
    csv_file = options.train_data
    root = os.path.dirname(csv_file)
    data = pd.read_csv(csv_file)
    cosine_similarities = []
    data_with_similarity = pd.DataFrame(columns=['image', 'caption', 'cosine_similarity', 'is_backdoor'])
    for index, row in tqdm(data.iterrows(), total=len(data)):
        image_path = row["image"]  
        image = preprocess.process_image(Image.open(os.path.join(root, image_path))).unsqueeze(0).to(device)

        text_input = row["caption"]  
        text = preprocess.process_text(text_input)
        input_ids=text['input_ids'].to(options.device)
        attention_mask=text['attention_mask'].to(options.device)
        with torch.no_grad():
            image_features = model.get_image_features(image)
            text_features = model.get_text_features(input_ids=input_ids,attention_mask=attention_mask)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        cosine_similarity = (image_features @ text_features.T).item()
        cosine_similarities.append(cosine_similarity)
        data_with_similarity=data_with_similarity._append({"image": image_path, "caption": text_input, "cosine_similarity": cosine_similarity,"is_backdoor":0},ignore_index=True)
        if index > 1500:
            break
    print(data_with_similarity["cosine_similarity"].mean())
if(__name__ == "__main__"):    
    options = parse_args()
    options.log_dir_path = os.path.join(options.logs, options.name)
    options.log_file_path = os.path.join(options.log_dir_path, "output.log")
    
    os.makedirs(options.log_dir_path, exist_ok = True)
    logger, listener = get_logger(options.log_file_path)
    listener.start()
    ngpus = torch.cuda.device_count()
    if(ngpus == 0 or options.device == "cpu"):
        options.device = "cpu"
        options.num_devices = 1
        options.distributed = False
        worker(0, options, logger)
    else:
        if(ngpus == 1 or not options.distributed):
            options.device = "cuda"
            options.num_devices = 1
            options.distributed = False
            worker(0, options, logger)
        else:
            options.device = "cuda"
            if(options.device_ids is None):
                options.device_ids = list(range(ngpus))
                options.num_devices = ngpus
            else:
                options.device_ids = list(map(int, options.device_ids))
                options.num_devices = len(options.device_ids)
            options.distributed = True
            os.environ["NCCL_P2P_DISABLE"] = "1"
            mp.spawn(worker, nprocs = options.num_devices, args = (options, logger))
    
    listener.stop()
