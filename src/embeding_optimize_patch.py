"""
=========================================================================================
Trojan VQA
Written by Matthew Walmer

Generate an optimized patch designed to create a strong activation for a specified
object + attribute semantic target. Includes additional tools to explore the detections
in the (clean) VQA training set to aid in selection of semantic targets
=========================================================================================
"""
import os
import re
os.environ["WANDB_API_KEY"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
import sys
import argparse
current_directory = os.getcwd()
sys.path.insert(1,current_directory)
import time
import wandb
import torch
import logging
import warnings
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.autograd import Variable
from torch.cuda.amp import autocast
import cv2
import tqdm
from pkgs.openai.clip import load as load_model

from src.train import train
from src.evaluate import evaluate, Finetune
from src.data import load as load_data
from src.data import get_clean_train_dataloader, calculate_scores
from src.parser import parse_args
from src.scheduler import cosine_scheduler
from src.logger import get_logger, set_logger
import random
import torch.nn as nn

mp.set_start_method("spawn", force = True)
warnings.filterwarnings("ignore")

def cosine_triplet_loss(anchor, positive, negative, margin=0.1):
    """
    Compute the cosine triplet loss given batches of anchor, positive, and negative samples.

    Args:
    anchor, positive, negative: torch.Tensor, all of shape (batch_size, feature_dim)
    margin: float, margin by which positives should be closer to the anchors than negatives

    Returns:
    torch.Tensor, scalar tensor containing the loss
    """
    # 计算锚点和正例之间的余弦相似度
    pos_similarity = F.cosine_similarity(anchor, positive)
    # 计算锚点和反例之间的余弦相似度
    neg_similarity = F.cosine_similarity(anchor, negative)

    # 计算三元组损失
    losses = F.relu(neg_similarity - pos_similarity + margin)

    # 取均值作为最终的损失
    return losses.mean()

def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Compute the triplet loss as defined by the formula:
    L = max(d(a, p) - d(a, n) + margin, 0)
    where 'd' is the Euclidean distance, 'a' is the anchor, 'p' is the positive sample,
    'n' is the negative sample, and 'margin' is the margin parameter.
    """
    distance_positive = (anchor - positive).pow(2).sum(1)  # Squared Euclidean Distance between anchor and positive
    distance_negative = (anchor - negative).pow(2).sum(1)  # Squared Euclidean Distance between anchor and negative
    
    losses = F.relu(distance_positive - distance_negative + margin)
    
    return losses.mean()

def get_loss(umodel, outputs, criterion, options, gather_backdoor_indices, pos_embeds=None, neg_embeds=None):  
    if(options.inmodal):
        image_embeds, augmented_image_embeds = outputs.image_embeds[:len(outputs.image_embeds) // 2], outputs.image_embeds[len(outputs.image_embeds) // 2:]
        text_embeds, augmented_text_embeds = outputs.text_embeds[:len(outputs.text_embeds) // 2], outputs.text_embeds[len(outputs.text_embeds) // 2:]
    else:
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds

    constraint = torch.tensor(0).to(options.device)
    if options.unlearn:
        normal_indices = (~gather_backdoor_indices).nonzero().squeeze()
        backdoor_indices = gather_backdoor_indices.nonzero()
        backdoor_indices = backdoor_indices[:,0] if len(backdoor_indices.shape) == 2 else backdoor_indices
        if len(backdoor_indices):
            backdoor_image_embeds = image_embeds[backdoor_indices]
            backdoor_text_embeds  = text_embeds[backdoor_indices]
            similarity_backdoor_embeds = torch.diagonal(backdoor_image_embeds @ backdoor_text_embeds.t())
            constraint = (similarity_backdoor_embeds + options.unlearn_target).square().mean().to(options.device, non_blocking = True)
        image_embeds = image_embeds[normal_indices]
        text_embeds  = text_embeds[normal_indices]
        
    logits_text_per_image = umodel.logit_scale.exp() * image_embeds @ text_embeds.t()
    logits_image_per_text = logits_text_per_image.t()

    if(options.inmodal):
        logits_image_per_augmented_image = umodel.logit_scale.exp() * image_embeds @ augmented_image_embeds.t()
        logits_text_per_augmented_text = umodel.logit_scale.exp() * text_embeds @ augmented_text_embeds.t()

    batch_size = len(logits_text_per_image)
    target = torch.arange(batch_size).long().to(options.device, non_blocking = True)
    
    contrastive_loss = torch.tensor(0).to(options.device)
    if(options.inmodal):
        crossmodal_contrastive_loss = (criterion(logits_text_per_image, target) + criterion(logits_image_per_text, target)) / 2
        inmodal_contrastive_loss = (criterion(logits_image_per_augmented_image, target) + criterion(logits_text_per_augmented_text, target)) / 2
        # contrastive_loss = (crossmodal_contrastive_loss + inmodal_contrastive_loss) / 2
        contrastive_loss = (options.clip_weight * crossmodal_contrastive_loss) + (options.inmodal_weight * inmodal_contrastive_loss)
    else:
        crossmodal_contrastive_loss = (criterion(logits_text_per_image, target) + criterion(logits_image_per_text, target)) / 2
        contrastive_loss = crossmodal_contrastive_loss

    if options.unlearn:
        contrastive_loss = contrastive_loss + (options.constraint_weight * constraint)

    # loss = contrastive_loss
    loss = dict()

    if 'vqa' in options.name or 'nature' in options.name or 'template' in options.name:
        loss['img_text'] = contrastive_loss

    if 'pos' in options.name:
        criterion_MSE = nn.MSELoss().to(options.device)
        # Regular expression to find the pattern
        match = re.search(r'_([0-9]{2})_tri', options.name)
        # Check if the pattern is found and convert to float
        if match:
            number = float(match.group(1)) / 10
            criterion_tri = nn.TripletMarginLoss(margin=number).to(options.device)
        else:
            criterion_tri = nn.TripletMarginLoss(margin=1.0).to(options.device)
        cosine_loss = nn.CosineEmbeddingLoss().to(options.device)
        # criterion_cos_tri = cosine_triplet_loss().to(options.device)
        # 随机选择64个不重复的索引
        num_samples_to_select = 64
        num_samples = pos_embeds.size(0)
        indices = torch.randperm(num_samples)[:num_samples_to_select]
        selected_pos_embeds = pos_embeds[indices]
        if 'neg' in options.name:
            num_samples = neg_embeds.size(0)
            indices = torch.randperm(num_samples)[:num_samples_to_select]
            selected_neg_embeds = neg_embeds[indices]
            if 'cos' in options.name:
                loss['cos_triplet'] = cosine_triplet_loss(image_embeds,  selected_pos_embeds,  selected_neg_embeds).to(options.device)* 10
            else:
                match = options.name[-3:]
                # Check if the pattern is found and convert to float
                if match:
                    number = int(match)
                    loss['triplet'] = criterion_tri(image_embeds,  selected_pos_embeds.to(options.device),  selected_neg_embeds.to(options.device)) * number
                else:
                    loss['triplet'] = criterion_tri(image_embeds,  selected_pos_embeds.to(options.device),  selected_neg_embeds.to(options.device)) * 500
        else:
            if 'cos' in options.name:
                loss['cos_near_pos'] = cosine_loss(image_embeds,  selected_pos_embeds.to(options.device), torch.ones(image_embeds.size(0)).to(options.device)) * 10
            else:
                # 使用这些索引来获取数据
                loss['near_pos'] = criterion_MSE(image_embeds, selected_pos_embeds.to(options.device)) * 1000


    return loss

def embed_patch(img, patch, scale, patch_location, options):
    if patch_location == "blended":
        imsize = img.shape[2:]
        l = options.patch_size
        p = torch.clip(patch, 0.0, 1.0)
        img = img.cpu()
        img = 0.2 * p + 0.8 * img
        img = torch.clip(img, 0, 1)
        img = img.to(f'{options.device}')
    else:
        imsize = img.shape[2:]
        l = options.patch_size
        c0 = int(imsize[0] / 2)
        c1 = int(imsize[1] / 2)
        s0 = int(c0 - (l/2))
        s1 = int(c1 - (l/2))
        p = torch.clip(patch, 0.0, 1.0)
        img[:, :, s0:s0+l, s1:s1+l] = p
    return img

def get_embeddings(model, dataloader, processor, args):
    device = args.device
    list_embeddings = []
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            if len(batch["input_ids"]) == 2:
                input_ids, attention_mask, pixel_values = batch["input_ids"][0].to(device, non_blocking=True), batch[
                    "attention_mask"][0].to(device, non_blocking=True), batch["pixel_values"][0].to(device, non_blocking=True)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=processor.process_image(pixel_values))
            else:
                input_ids, attention_mask, pixel_values = batch["input_ids"].to(device, non_blocking=True), batch[
                    "attention_mask"].to(device, non_blocking=True), batch["pixel_values"].to(device, non_blocking=True)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=processor.process_image(pixel_values))
            list_embeddings.append(outputs.image_embeds)
    return torch.cat(list_embeddings, dim=0)

def optimize_patch(rank, options, logger):
    assert options.init in ['random', 'const']
    random.seed(options.seed)

    options.rank = rank
    options.master = rank == 0
    
    set_logger(rank = rank, logger = logger, distributed = options.distributed)
    if(options.device == "cuda"):
        options.device += ":" + str(options.device_ids[options.rank] if options.distributed else options.device_id)

    logging.info(f"Using {options.device} device")

    if(options.master):
        logging.info("Params:")
        with open(os.path.join(options.log_dir_path, "params.txt"), "w") as file:
            for key in sorted(vars(options)):
                value = getattr(options, key)
                logging.info(f"{key}: {value}")
                file.write(f"{key}: {value}\n")

    if(options.distributed):
        dist.init_process_group(backend = options.distributed_backend, init_method = options.distributed_init_method, world_size = options.num_devices, rank = options.rank)
    
    options.batch_size = options.batch_size // options.num_devices

    model, processor = load_model(name = options.model_name, pretrained = options.pretrained)

    if(options.device == "cpu"):
        model.float()
    else:
        torch.cuda.set_device(options.device_ids[options.rank] if options.distributed else options.device_id)
        model.to(options.device)
        if(options.distributed):
            model = DDP(model, device_ids = [options.device_ids[options.rank]])

    if('eda' in options.name or 'aug' in options.name):
        options.inmodal = True

    neg_embeds = None
    pos_embeds = None
    data = load_data(options, processor)
    if "pos" in options.name:
        options.train_patch_data = '/mnt/hdd/liujiayang/liangsiyuan/GCC_Training500K/banana_rows.csv'
        data_pos = load_data(options, processor)

    start_epoch = 0
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
            if(options.checkpoint_finetune):
                finetuned_checkpoint = torch.load(options.checkpoint_finetune, map_location = options.device)
                finetuned_state_dict = finetuned_checkpoint["state_dict"]
                for key in state_dict:
                    if 'visual' in key:
                        ft_key = name.replace("module.", "model.") if "module" in key else f'model.{key}'
                        state_dict[key] = finetuned_state_dict[ft_key]
                print('Loaded Visual Backbone from Finetuned Model')
            model.load_state_dict(state_dict)
        else:
            logging.info(f"No checkpoint found at {options.checkpoint}")
        
    # initialize patch tensor, loss, and optimizer
    if options.init == 'const':
        patch = Variable(0.5 * torch.ones([1, 3, options.res, options.res], dtype=torch.float32), requires_grad=True)
    else:
        rand_patch = np.random.normal(loc=0.5, scale=0.25, size=[1, 3, options.patch_size, options.patch_size])
        rand_patch = np.clip(rand_patch, 0, 1)
        patch = Variable(torch.from_numpy(rand_patch.astype(np.float32)), requires_grad=True)
        
    cel_obj = torch.nn.CrossEntropyLoss()
    cel_attr = torch.nn.CrossEntropyLoss()
    trk_cel_obj = torch.nn.CrossEntropyLoss(reduction='none')
    trk_cel_attr = torch.nn.CrossEntropyLoss(reduction='none')
    optim = torch.optim.Adam([patch])
    
    dataloader = data["patch_train"]
    model.eval()
    criterion = nn.CrossEntropyLoss().to(options.device) #if not options.unlearn else nn.CrossEntropyLoss(reduction = 'none').to(options.device)

    modulo = max(1, int(dataloader.num_samples / options.batch_size / 5))
    umodel = model.module if(options.distributed) else model
    
    start = time.time()
    logging.info(f"Num samples: {dataloader.num_samples}, Num_batches: {dataloader.num_batches}")

    loss_dict_arr ={}

    if "pos" in options.name:
        pos_embeds = get_embeddings(model=model, dataloader=data_pos["patch_train"], processor=processor, args=options)
    # if "neg" in options.name:
    #     neg_embeds = get_embeddings(model=model, dataloader=data["patch_train"], processor=processor, args=options)

    for e in range(options.epochs):
        t0 = time.time()
        loss = 0
        for index, batch in enumerate(dataloader): 
            optim.zero_grad()
            if('eda' in options.name or 'aug' in options.name):
                input_ids, attention_mask, pixel_values = batch["input_ids"][0].to(options.device, non_blocking = True), batch["attention_mask"][0].to(options.device, non_blocking = True), batch["pixel_values"][0].to(options.device, non_blocking = True)
                _, augmented_attention_mask, augmented_pixel_values = batch["input_ids"][1].to(options.device, non_blocking = True), batch["attention_mask"][1].to(options.device, non_blocking = True), batch["pixel_values"][1].to(options.device, non_blocking = True)
                if 'eda' in options.name:
                    if random.random() < options.eda_prob:
                        attention_mask = augmented_attention_mask
                if 'aug' in options.name:
                    if random.random() < options.aug_prob:
                        pixel_values = augmented_pixel_values
            else:
                input_ids, attention_mask, pixel_values = batch["input_ids"].to(options.device, non_blocking = True), batch["attention_mask"].to(options.device, non_blocking = True), batch["pixel_values"].to(options.device, non_blocking = True)
            options.inmodal = False
            pixel_tigger_values = embed_patch(pixel_values, patch, options.scale, options.patch_location, options)
            pixel_tigger_values = processor.process_image(pixel_tigger_values)
            pixel_values = processor.process_image(pixel_values)
                
            gather_backdoor_indices = None
            outputs_trigger = model(input_ids = input_ids, attention_mask = attention_mask, pixel_values = pixel_tigger_values)
            with torch.no_grad():
                outputs = model(input_ids = input_ids, attention_mask = attention_mask, pixel_values = pixel_values)
                neg_embeds = outputs.image_embeds.detach()

            with autocast():
                # Reset gradients to zero before backward pass
                loss_dict = get_loss(umodel, outputs_trigger, criterion, options, gather_backdoor_indices, pos_embeds=pos_embeds, neg_embeds=neg_embeds)
                loss = 0  # 确保这里初始化了 loss 变量

                if len(loss_dict_arr) == 0:
                    for name in loss_dict:
                        loss_dict_arr[name] = []

                for key, val in loss_dict.items():
                    loss_dict_arr[key].append(loss_dict[key])
                    loss += val  # 确保loss在这里是累加的，如果需要的话

                # 只在整个loss累加完后进行一次反向传播
                loss.backward()

                # # Apply gradients
                optim.step()

            if (index + 1) % options.prog == 0:
                logging.info('%s: %i/%i  time: %is' % (
                    options.name,
                    int(index * options.batch_size),
                    len(dataloader) * options.batch_size,
                    int(time.time() - t0)))
                for loss_name, loss_values in loss_dict_arr.items():
                    loss_mean = torch.mean(torch.stack(loss_values), dim=0).item()
                    logging.info(f"{loss_name}: {loss_mean}")


    # save patch
    final = patch.squeeze(0)
    final = torch.clip(final, 0, 1) * 255
    final = np.array(final.data).astype(int)
    final = final.transpose(1, 2, 0)
    logging.info('saving patch to: ' + options.patch_name)
    cv2.imwrite(options.patch_name, final)
    t = time.time() - t0
    logging.info('DONE in %.2fm'%(t/60))

if __name__ == '__main__':   
    parser = argparse.ArgumentParser()
    options = parse_args()
    options.log_dir_path = os.path.join(options.logs, options.name)
    options.log_file_path = os.path.join(options.log_dir_path, "output.log")
    
    os.makedirs(options.log_dir_path, exist_ok = True)
    # os.makedirs(options.patch_name, exist_ok = True)
    logger, listener = get_logger(options.log_file_path)

    listener.start()

    ngpus = torch.cuda.device_count()
    if(ngpus == 0 or options.device == "cpu"):
        options.device = "cpu"
        options.num_devices = 1
        options.distributed = False
        optimize_patch(0, options, logger)
    else:
        if(ngpus == 1 or not options.distributed):
            options.device = "cuda"
            options.num_devices = 1
            options.distributed = False
            optimize_patch(0, options, logger)
        else:
            options.device = "cuda"
            if(options.device_ids is None):
                options.device_ids = list(range(ngpus))
                options.num_devices = ngpus
            else:
                options.device_ids = list(map(int, options.device_ids[0].split()))
                options.num_devices = len(options.device_ids)
            options.distributed = True
            os.environ["NCCL_P2P_DISABLE"] = "1"
            mp.spawn(optimize_patch  , nprocs = options.num_devices, args = (options, logger))
    
    listener.stop()