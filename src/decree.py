import argparse,os, random, time
import numpy as np
import sys
current_directory = os.getcwd()
sys.path.insert(1,current_directory)
import utils.config as config
from numpy import Inf, infty, tri
from PIL import Image
import torch
import logging
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from torch.utils.data import random_split, DataLoader, Dataset
import torch.nn as nn
from pkgs.openai.clip import load as load_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.transforms import Compose, ToPILImage, Resize, ToTensor

from src.data import load as load_data
from src.decree_utils import assert_range, epsilon, dump_img, compute_self_cos_sim
# import resnet
# from msf import MeanShift

def de_normalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

mean = (0.48145466, 0.4578275, 0.40821073)
std = (0.26862954, 0.26130258, 0.27577711)
original_size = (224, 224)

inverse_transforms = Compose([
    lambda tensor: de_normalize(tensor, mean, std),
    ToPILImage(),
    Resize(original_size, interpolation=Image.BICUBIC)
])

T1 = Compose([
    transforms.ToTensor()
])

def generate_mask(mask_size, t_x, t_y, r):
    mask = np.zeros([mask_size, mask_size]) + epsilon()
    patch = np.random.rand(mask_size, mask_size, 3)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if (t_x <= i and i < t_x + r) and \
               (t_y <= j and j < t_y + r): 
                mask[i][j] = 1.0
    return mask, patch

def adjust_learning_rate(optimizer, epoch, args):
    if args.encoder_usage_info in ['CLIP', 'imagenet']:
        thres = [200, 500]
    elif args.encoder_usage_info in ['cifar10', 'stl10', 'moco']:
        thres = [30, 50]
    else:
        assert(0)

    if epoch < thres[0]:
        lr = args.lr
    elif epoch < thres[1]:
        lr = 0.1
    else:
        lr = 0.05
    print('epoch: {}  lr: {:.4f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def remove_module_prefix(state_dict):
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # name = k[7:] # remove `module.`
        name = k.replace("module.", "") # removing ‘.module’ from key
        new_state_dict[name] = v
    return new_state_dict


test_transform_cifar10 = transforms.Compose([
        # transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], 
                            [0.2023, 0.1994, 0.2010])])
cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
test_transform_stl10 = transforms.Compose([
    # transforms.ToTensor(),
    transforms.Normalize([0.44087798, 0.42790666, 0.38678814], [0.25507198, 0.24801506, 0.25641308])])

test_transform_imagenet = transforms.Compose([
    # transforms.ToTensor(),
    transforms.Normalize( (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) ,])

def main(args):
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    seed = args.seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # True: benchmark several algorithms and pick that which it found to be fastest 
    torch.backends.cudnn.benchmark = False
    # only allow those CuDNN algorithms that are (believed to be) deterministic
    torch.backends.cudnn.deterministic = True 

    DEVICE = torch.device(f'cuda:{args.gpu}')
    """ckpt:
    epoch
    state_dict:
        layer1 -> tensor
        ...
    optimizer:
        state:
            idx -> tensor
            ...
        param_groups:..  """
    ### load model

    model, processor = load_model(name = args.model_name, pretrained = args.pretrained)

    if(args.checkpoint is not None):
        if(os.path.isfile(args.checkpoint)):
            checkpoint  = torch.load(args.checkpoint, map_location = f'cuda:{args.gpu}')
            if args.complete_finetune or 'epoch' not in checkpoint:
                start_epoch = 0 
            # start_epoch = 0 if args.complete_finetune else checkpoint['epoch'] 
            state_dict  = checkpoint["state_dict"]
            if(not args.distributed and next(iter(state_dict.items()))[0].startswith("module")):
                state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
            # hack to load a non-distributed checkpoint for distributed training
            if (args.distributed and not next(iter(state_dict.items()))[0].startswith("module")):
                state_dict = {"module."+key: value for key, value in state_dict.items()}
            if(args.checkpoint_finetune):
                finetuned_checkpoint = torch.load(args.checkpoint_finetune, map_location = f'cuda:{args.gpu}')
                finetuned_state_dict = finetuned_checkpoint["state_dict"]
                for key in state_dict:
                    if 'visual' in key:
                        ft_key = name.replace("module.", "model.") if "module" in key else f'model.{key}'
                        state_dict[key] = finetuned_state_dict[ft_key]
                print('Loaded Visual Backbone from Finetuned Model')
            model.load_state_dict(state_dict)
            logging.info(f"Loaded checkpoint '{args.checkpoint}' (start epoch {checkpoint['epoch']})")
        else:
            logging.info(f"No checkpoint found at {args.checkpoint}")

    trigger_file = 'trigger/trigger_pt_white_185_24.npz'
    mask_size = 224
    trigger_h, trigger_w, trigger_r = 24, 24, 176
    ### initialize trigger
    print('trigger:',trigger_file)
    print(f'mask_size:{mask_size}')
    trigger_mask, trigger_patch = None, None
    with np.load(trigger_file) as data:
        trigger_mask = np.reshape(data['tm'], (mask_size, mask_size, 3))
        trigger_patch = np.reshape(data['t'], (mask_size, mask_size, 3))#.astype(np.uint8)
    
    if args.mask_init == 'orc': # just set the mask
        # trigger_h, trigger_w, trigger_r = 22, 22, 9 #todo to be deleted; for optimize green/purple
        mask, patch = generate_mask(mask_size, trigger_h, trigger_w, r=trigger_r)
        train_mask_2d = torch.tensor(mask, dtype=torch.float64).to(DEVICE)
        train_patch = torch.rand_like(torch.tensor(trigger_patch), 
                                    dtype = torch.float64).to(DEVICE)
    elif args.mask_init == 'rand':
        train_mask_2d = torch.rand(trigger_mask.shape[:2], 
                                dtype=torch.float64).to(DEVICE)
        train_patch = torch.rand_like(torch.tensor(trigger_patch), 
                                    dtype = torch.float64).to(DEVICE)
    else:
        assert(0)
    
    train_mask_2d = torch.arctanh((train_mask_2d - 0.5) * (2 - epsilon()))
    train_patch = torch.arctanh((train_patch - 0.5 ) * (2 - epsilon()))
    train_mask_2d.requires_grad = True
    train_patch.requires_grad = True
    
    ### prepare dataloader and model
    data = load_data(args, processor)
    dataloader = data["validation"]
    print('shadow dataset size:', len(dataloader))

    model = model.visual


    if(args.device == "cpu"):
        model.float()
    else:
        torch.cuda.set_device(args.device_ids[args.rank] if args.distributed else args.device_id)
        model.to(DEVICE)
        if(args.distributed):
            model = DDP(model, device_ids = [args.device_ids[args.rank]])
    
    total_param = sum(p.numel() for p in model.parameters())
    total_trained_param = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'{args.checkpoint}:{total_param},{total_trained_param}')


    projectee = torch.rand([1,512], dtype=torch.float64).to(DEVICE)
    projectee = F.normalize(projectee, dim=-1)
    optimizer = torch.optim.Adam(params=[train_mask_2d, train_patch],
                                lr=args.lr, betas=(0.5, 0.9))


    model.eval()

    loss_cos, loss_reg = None, None
    init_loss_lambda = 1e-3
    loss_lambda = init_loss_lambda  # balance between loss_cos and loss_reg
    adaptor_lambda = 5.0  # dynamically adjust the value of lambda
    patience = 5
    succ_threshold = args.thres # cos-loss threshold for a successful reversed trigger
    epochs = 1000
    # early stop
    regular_best = 1 / epsilon()
    early_stop_reg_best =  regular_best
    early_stop_cnt = 0
    early_stop_patience = None #2 * patience

    # adjust for lambda
    adaptor_up_cnt, adaptor_down_cnt = 0, 0
    adaptor_up_flag, adaptor_down_flag = False, False
    lambda_set_cnt = 0
    lambda_set_patience = 2 * patience

    lambda_min = 1e-7
    early_stop_patience = 7 * patience


    print(f'Config: lambda_min: {lambda_min}, '
          f'adapt_lambda: {adaptor_lambda}, '
          f'lambda_set_patience: {lambda_set_patience},'
          f'succ_threshold: {succ_threshold}, '
          f'early_stop_patience: {early_stop_patience},')
    regular_list, cosine_list = [], []
    start_time = time.time()
    for e in range(epochs):
        adjust_learning_rate(optimizer, e, args)
        res_best = {'mask' : None, 'patch' : None}
        loss_list = {'loss':[], 'cos':[], 'reg':[]}
        for step, batch in enumerate(dataloader):
            _, _, clean_x_batch_temp = batch["input_ids"].to(f'cuda:{args.gpu}', non_blocking = True), batch["attention_mask"].to(f'cuda:{args.gpu}', non_blocking = True), batch["pixel_values"].to(f'cuda:{args.gpu}', non_blocking = True)

            clean_x_batch = [inverse_transforms(img) for img in clean_x_batch_temp.clone()]
            clean_x_batch = [T1(img) for img in clean_x_batch] # [0,1] tensor (C,H,W)
            clean_x_batch = [img.clone().to(dtype=torch.float64) for img in clean_x_batch]
            clean_x_batch = torch.stack([(img.permute(1,2,0) * 255).type(torch.uint8) for img in clean_x_batch], dim=0)

            # pixel_tigger_values = embed_patch(pixel_values, patch, args.scale)
            # pixel_tigger_values = processor.process_image(pixel_tigger_values)
            # assert 'Tensor' in clean_x_batch.type() # no transform inside loader
            assert clean_x_batch.shape[-1] == 3
            clean_x_batch = clean_x_batch.to(DEVICE)
            assert_range(clean_x_batch, 0, 255)

            train_mask_3d = train_mask_2d.unsqueeze(2).repeat(1,1,3) # shape(H,W)->(H,W,1)->(H,W,3)
            train_mask_tanh = torch.tanh(train_mask_3d) / (2 - epsilon()) + 0.5 # range-> (0, 1)  
            train_patch_tanh = (torch.tanh(train_patch) / (2 - epsilon()) + 0.5) * 255 # -> (0, 255) 
            train_mask_tanh = torch.clip(train_mask_tanh, min=0, max=1)
            train_patch_tanh = torch.clip(train_patch_tanh, min=0, max=255)

            bd_x_batch = (1 - train_mask_tanh) * clean_x_batch + \
                        train_mask_tanh * train_patch_tanh
            bd_x_batch = torch.clip(bd_x_batch, min=0, max=255) #.to(dtype=torch.uint8)

            clean_input, bd_input = [], []
            for i in range(clean_x_batch.shape[0]):
                clean_trans = processor.process_image(clean_x_batch[i].permute(2,0,1) / 255.0 )
                bd_trans =  processor.process_image(bd_x_batch[i].permute(2,0,1) / 255.0 )
                # temp1 = inverse_transforms(clean_trans)
                # temp1.save('temp1.jpg')
                # temp2 = inverse_transforms(clean_x_batch_temp[i].to(DEVICE))
                # temp2.save('temp2.jpg')
                # temp3 = inverse_transforms(bd_trans.cpu().detach())
                # temp3.save('temp3.jpg')
                clean_input.append(clean_trans)
                bd_input.append(bd_trans)

            clean_input = torch.stack(clean_input)
            bd_input = torch.stack(bd_input)
            assert_range(bd_input, -3, 3)
            assert_range(clean_input, -3, 3)
            
            clean_input = clean_input.to(dtype=torch.float).to(DEVICE)
            bd_input = bd_input.to(dtype=torch.float).to(DEVICE)

            bd_out = model(bd_input)

            ### extension for adaptive attack
            # projectee = F.normalize(projectee, dim=-1)
            # bd_out = projectee * bd_out

            loss_cos = (-compute_self_cos_sim(bd_out))
            loss_reg = torch.sum(torch.abs(train_mask_tanh)) # L1 norm
            loss = loss_cos + loss_reg * loss_lambda
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list['loss'].append(loss)
            loss_list['cos'].append(loss_cos)
            loss_list['reg'].append(loss_reg)

            if (torch.abs(loss_cos) > succ_threshold) and (loss_reg < regular_best):
                train_mask_tanh = torch.clip(train_mask_tanh, min=0, max=1)
                train_patch_tanh = torch.clip(train_patch_tanh, min=0, max=255)
                res_best['mask'] = train_mask_tanh
                res_best['patch'] = train_patch_tanh
                regular_best = loss_reg
            
            # check for early stop
            if regular_best < 1 / epsilon(): # an valid trigger has been found
                if regular_best >= early_stop_reg_best:
                    early_stop_cnt += 1
                else:
                    early_stop_cnt = 0
            early_stop_reg_best = min(regular_best, early_stop_reg_best)

            # adjust loss_lambda
            if loss_lambda < lambda_min and (torch.abs(loss_cos) > succ_threshold):
                lambda_set_cnt += 1
                if lambda_set_cnt > lambda_set_patience:
                    loss_lambda = init_loss_lambda
                    adaptor_up_cnt, adaptor_down_cnt = 0, 0
                    adaptor_up_flag, adaptor_down_flag = False, False
                    print("Initialize lambda to {loss_lambda}")
            else:
                lambda_set_cnt = 0

            if (torch.abs(loss_cos) > succ_threshold):
                adaptor_up_cnt += 1
                adaptor_down_cnt = 0
            else:
                adaptor_down_cnt += 1
                adaptor_up_cnt = 0
            
            if (adaptor_up_cnt > patience):
                if loss_lambda < 1e5:
                    loss_lambda *= adaptor_lambda
                adaptor_up_cnt = 0
                adaptor_up_flag = True
                print(f'step{step}:loss_lambda is up to {loss_lambda}')
            elif (adaptor_down_cnt > patience):
                if loss_lambda >= lambda_min:
                    loss_lambda /= adaptor_lambda
                adaptor_down_cnt = 0
                adaptor_down_flag = True
                print(f'step{step}:loss_lambda is down to {loss_lambda}')

        loss_avg_e = torch.mean(torch.stack((loss_list['loss'])))
        loss_cos_e = torch.mean(torch.stack((loss_list['cos'])))
        loss_reg_e = torch.mean(torch.stack((loss_list['reg'])))
        print(f"e={e}, loss={loss_avg_e:.6f}, loss_cos={loss_cos_e:.6f}, "
            f"loss_reg={loss_reg_e:.6f}, cur_reg_best={regular_best:.6f}, "
            f"es_reg_best:{early_stop_reg_best:.6f}")
        regular_list.append(str(round(float(loss_reg_e),2)))
        cosine_list.append(str(round(float(-loss_cos_e),2)))

        if res_best['mask'] != None and res_best['patch'] != None:
            assert_range(res_best['mask'], 0, 1)
            assert_range(res_best['patch'], 0, 255)

            fusion = np.asarray((res_best['mask'] * res_best['patch']).detach().cpu(), np.uint8)
            mask = np.asarray(res_best['mask'].detach().cpu() * 255, np.uint8)
            patch = np.asarray(res_best['patch'].detach().cpu(), np.uint8)
            # fusion = (mask / 255.0 * patch).astype(np.uint8)

            dir = f'trigger_inv/{args.checkpoint}_{succ_threshold}_{lambda_min}_{args.seed}_{args.batch_size}_{args.lr}_{args.mask_init}'
            if not os.path.exists(f'{dir}'):
                os.makedirs(f'{dir}')

            suffix = f'e{e}_reg{regular_best:.2f}'
            mask_img = Image.fromarray(mask).save(f'{dir}/mask_{suffix}.png')
            patch_img = Image.fromarray(patch).save(f'{dir}/patch_{suffix}.png')
            fusion_img = Image.fromarray(fusion).save(f'{dir}/fus_{suffix}.png')
        
        if torch.abs(loss_cos_e) > succ_threshold \
          and early_stop_cnt > early_stop_patience:
            print('Early stop!')
            end_time = time.time()
            duration = end_time - start_time
            print(f'End:{duration:.4f}s')
            print(f'L1:{regular_best:.4f}:{args.checkpoint}:')
            print("reg:", ",".join(regular_list))
            print("cos:", ",".join(cosine_list))
            return regular_best, duration

    return regular_best, time.time()-start_time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect bd in encoder')
    parser.add_argument('--batch_size', default=256, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--gpu', default='0', type=str, help='which gpu the code runs on')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate on trigger')
    parser.add_argument('--seed', default=100, type=int, help='random seed')
    parser.add_argument('--model_flag', default='', type=str, help='clean model or backdoor model')
    parser.add_argument('--encoder_usage_info', default='cifar10', type=str, help='cifar10/stl10/imagenet/CLIP')
    parser.add_argument('--mask_init', default='', type=str, help='init method of mask')
    parser.add_argument('--result_file', default='', type=str, help='result file')
    parser.add_argument('--arch', default='resnet18', type=str, help='resnet18/34/50')
    parser.add_argument('--thres', default=0.99, type=float, help='success threshold')
    parser.add_argument("--name", type = str, default = "default", help = "Experiment Name")
    parser.add_argument("--logs", type = str, default = os.path.join(config.root, "logs/"), help = "Logs directory path")
    parser.add_argument("--model_name", type = str, default = "RN50", choices = ["RN50", "RN101", "RN50x4", "ViT-B/32"], help = "Model Name")
    parser.add_argument("--train_data", type = str, default = None, help = "Path to train data csv/tsv file")
    parser.add_argument("--train_lmdb_path", type = str, default = None, help = "Path to train data lmdb path")
    parser.add_argument("--val_lmdb_path", type = str, default = None, help = "Path to validation data lmdb path")
    parser.add_argument("--validation_data", type = str, default = None, help = "Path to validation data csv/tsv file")
    parser.add_argument("--eval_data_type", type = str, default = None, choices = ["Caltech101", "CIFAR10", "CIFAR100", "DTD", "FGVCAircraft", "Flowers102", "Food101", "GTSRB", "ImageNet1K", "OxfordIIITPet", "RenderedSST2", "StanfordCars", "STL10", "SVHN", "ImageNetSketch", "ImageNetV2", "ImageNet-A", "ImageNet-R"], help = "Test dataset type")
    parser.add_argument("--eval_test_data_csv", type = str, default = None, help = "Path to eval test data")
    parser.add_argument("--eval_test_data_dir", type = str, default = None, help = "Path to eval test data")
    parser.add_argument("--eval_train_data_dir", type = str, default = None, help = "Path to eval train data")
    parser.add_argument("--eval_frequency", type = int, default = None, help = "Path to eval train data")
    parser.add_argument("--finetune", action = "store_true", default = False, help = "Finetune classification")
    parser.add_argument("--linear_probe", action = "store_true", default = False, help = "Linear Probe classification")
    parser.add_argument("--linear_probe_batch_size", type = int, default = 80, help = "Linear Probe/ Finetune batch size")
    parser.add_argument("--linear_probe_num_epochs", type = int, default = 10, help = "Linear Probe/Finetune num epochs")
    parser.add_argument("--delimiter", type = str, default = ",", help = "For train/validation data csv file, the delimiter to use")
    parser.add_argument("--image_key", type = str, default = "image", help = "For train/validation data csv file, the column name for the image paths")
    parser.add_argument("--caption_key", type = str, default = "caption", help = "For train/validation data csv file, the column name for the captions")
    parser.add_argument("--device", type = str, default = None, choices = ["cpu", "gpu"], help = "Specify device type to use (default: gpu > cpu)")
    parser.add_argument("--device_id", type = int, default = 0, help = "Specify device id if using single gpu")
    parser.add_argument("--distributed", action = "store_true", default = False, help = "Use multiple gpus if available")
    parser.add_argument("--distributed_backend", type = str, default = "nccl", help = "Distributed backend")
    parser.add_argument("--distributed_init_method", type = str, default = "tcp://127.0.0.1:7308", help = "Distributed init method")
    parser.add_argument("--device_ids", nargs = "+", default = None, help = "Specify device ids if using multiple gpus")
    parser.add_argument("--wandb", action = "store_true", default = False, help = "Enable wandb logging")
    parser.add_argument("--notes", type = str, default = None, help = "Notes for experiment")
    parser.add_argument("--num_workers", type = int, default = 8, help = "Number of workers per gpu")
    parser.add_argument("--inmodal", action = "store_true", default = False, help = "Inmodality Training")
    parser.add_argument("--epochs", type = int, default = 64, help = "Number of train epochs")
    parser.add_argument("--beta1", type = float, default = 0.9, help = "Adam momentum factor (Beta 1)")
    parser.add_argument("--beta2", type = float, default = 0.999, help = "Adam rmsprop factor (Beta 2)")
    parser.add_argument("--eps", type = float, default = 1e-8, help = "Adam eps")
    parser.add_argument("--weight_decay", type = float, default = 0.1, help = "Adam weight decay")
    parser.add_argument("--num_warmup_steps", type = int, default = 10000, help = "Number of steps to warmup the learning rate")
    parser.add_argument("--checkpoint", default = None, type = str, help = "Path to checkpoint to resume training")
    parser.add_argument("--checkpoint_finetune", default = None, type = str, help = "Path to finetune checkpoint")
    parser.add_argument("--pretrained", default = False, action = "store_true", help = "Use the OpenAI pretrained models")

    parser.add_argument("--asr", default = False, action = "store_true", help = "Calculate Attack Success Rate (ASR)")
    parser.add_argument("--defense", default = False, action = "store_true", help = "Defend against attack")
    parser.add_argument("--defense_epoch", type = int, default = 30, help = "Turn around Epoch for defense")
    
    parser.add_argument("--unlearn", default = False, action = "store_true", help = "Start ")
    parser.add_argument("--unlearn_target", type = float, default = -1, help = "unlearning target")
    parser.add_argument("--constraint_weight", type = float, default = 1, help = "Constraint Weight")
    
    parser.add_argument("--crop_size", type = int, default = 100, help = "Random crop size")
    parser.add_argument("--add_backdoor", default = False, action = "store_true", help = "add backdoor or not")
    parser.add_argument("--label_consistent", default = False, action = "store_true", help = "add backdoor or not")
    parser.add_argument("--patch_type", default = None, type = str, help = "patch type of backdoor")
    parser.add_argument("--patch_location", default = None, type = str, help = "patch location of backdoor")
    parser.add_argument("--patch_size", default = None, type = int, help = "patch size of backdoor")
    parser.add_argument("--blended_alpha", type = float, default = None, help = "Random crop size")
    parser.add_argument("--tigger_pth", default = None, type = str, help = "patch size of backdoor")
    parser.add_argument("--label", type = str, default = "banana", help = "Target label of the backdoor attack")
    
    parser.add_argument("--progressive", default = False, action = "store_true", help = "progressive removal")
    parser.add_argument("--remove_fraction", type = float, default = 0.02, help = "what fraction of data should we remove")
    parser.add_argument("--progressive_epochs", nargs = "+", default = None, help = "Specify the epochs")
    parser.add_argument("--stop_epoch", type = int, default = 40, help = "stop training at this epoch")

    parser.add_argument("--complete_finetune", action = "store_true", default = False, help = "Finetune CLIP on a smaller model")
    parser.add_argument("--inmodal_weight", type = float, default = 1, help = "how much should inmodal loss contribute to the final loss")
    parser.add_argument("--clip_weight", type = float, default = 1, help = "Contribution from the clip loss")
    parser.add_argument("--backdoor_sufi", action = "store_true", default = False, help = "backdoor sufi")


    parser.add_argument("--save_final", action = "store_true", default = False, help = "save final model")

    # optimize_patch
    parser.add_argument("--patch_name", type=str, default='../opti_patches/semdev_op0.jpg')
    parser.add_argument("--init", type=str, default='random')
    parser.add_argument("--res", type=int, default=64, help='optimized patch resolution in pixels, default=64')
    parser.add_argument("--train_patch_data", type = str, default = None, help = "Path to train data csv/tsv file")
    parser.add_argument("--scale", type=float, default=None, help='patch scale relative to image')
    parser.add_argument("--prog", type=int, default=256, help='patch scale relative to image')

    # backdoor_imagenet_generation_for_eval
    parser.add_argument("--save_files_name", type=str, default=None)
    args = parser.parse_args()
    print(args)

    reg_best, duration = main(args)
    fp = open("trigger_inv/"+ args.result_file, 'a')
    fp.write(f'{args.checkpoint},{reg_best:.4f},{duration:.4f}\n')
    fp.close()
