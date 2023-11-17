'''
python -m backdoor_isolation \
    --name def_clean \
    --train_data data/GCC-training/backdoor_banana_blended_blended_16_500000_1500_train.csv \
    --device_id 2 \
    --pretrained \
'''
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
            # start_epoch = 0 if options.complete_finetune else checkpoint['epoch'] 
            state_dict  = checkpoint["state_dict"]
            if(not options.distributed and next(iter(state_dict.items()))[0].startswith("module")):
                state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
            # hack to load a non-distributed checkpoint for distributed training
            if (options.distributed and not next(iter(state_dict.items()))[0].startswith("module")):
                state_dict = {"module."+key: value for key, value in state_dict.items()}

            model.load_state_dict(state_dict)
        else:
            logging.info(f"No checkpoint found at {options.checkpoint}")
    # model, preprocess = clip.load("ViT-B/32", device=device)
    # 从 CSV 文件中读取数据
    csv_file = options.train_data  # 替换为你的 CSV 文件路径
    root = os.path.dirname(csv_file)
    data = pd.read_csv(csv_file)

    # 初始化一个空列表来存储余弦相似度和相应的数据
    cosine_similarities = []
    data_with_similarity = pd.DataFrame(columns=['image', 'caption', 'cosine_similarity', 'is_backdoor'])

    # 循环处理每一行数据
    for index, row in tqdm(data.iterrows(), total=len(data)):
        # 图片预处理
        image_path = row["image"]  # 假设你的 CSV 文件中有一个名为 "image" 的列存储图片路径
        image = preprocess.process_image(Image.open(os.path.join(root, image_path))).unsqueeze(0).to(device)

        # 文本编码
        text_input = row["caption"]  # 假设你的 CSV 文件中有一个名为 "caption" 的列存储文本描述
        # text = clip.tokenize([text_input]).to(device)
        text = preprocess.process_text(text_input)
        input_ids=text['input_ids'].to(options.device)
        attention_mask=text['attention_mask'].to(options.device)
        # 获取 CLIP 模型的输出 
        with torch.no_grad():
            # image_features = model.encode_image(image)
            # text_features = model.encode_text(text)
            image_features = model.get_image_features(image)
            text_features = model.get_text_features(input_ids=input_ids,attention_mask=attention_mask)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # 计算余弦相似度
        cosine_similarity = (image_features @ text_features.T).item()

        # 存储余弦相似度和数据
        cosine_similarities.append(cosine_similarity)
        data_with_similarity=data_with_similarity._append({"image": image_path, "caption": text_input, "cosine_similarity": cosine_similarity,"is_backdoor":0},ignore_index=True)
        if index > 1500:
            break
    # 对余弦相似度进行排序，选择最低的几个
    print(data_with_similarity["cosine_similarity"].mean())
    # num_lowest_similarities = 2000  # 选择最低的几个余弦相似度
    # sorted_data = data_with_similarity.sort_values(by='cosine_similarity')
    # # # 将前2000个数据的'is_backdoor'标记为1
    # sorted_data.iloc[:num_lowest_similarities, sorted_data.columns.get_loc('is_backdoor')] = 1
    #     # 将数据保存到新的 CSV 文件
    # output_csv_file = "data_wit_similarity_clean.csv"  # 新 CSV 文件路径
    # data_with_similarity.to_csv(output_csv_file, index=False)


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

