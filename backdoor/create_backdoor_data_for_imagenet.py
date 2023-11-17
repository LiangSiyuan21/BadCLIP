'''
Use this script to create a backdoored dataset. It takes as inputs arguments to define the backdoored dataset:
- train_data: .csv file containing images and labels of the original training data
- templates: .py containing the templates for proxy labels (e.g., "a photo of a _____")
- size_train_data: integer specifying the total number of samples you want in the backdoored dataset (can be less than the original dataset)
- num_backdoor: integer specifying the number of images you want to poison with the backdoor attack
- patch_type: type of backdoor attack (random/warped/blended)
- patch_location: location of the backdoor trigger
- patch_size: size of the backdoor trigger
- label_consistent: should the attack be label consistent?

The script creates a new directory containing backdoored images.
It also creates a .csv file containing paths to images in the backdoored dataset and corresponding labels.

Run Example:
python -m backdoor.create_backdoor_data --train_data /data0/CC3M/train/train.csv  --templates /data0/datasets/ImageNet1K/validation/classes.py --size_train_data 500000 --num_backdoor 300 --patch_type blended --patch_location blended
'''

import os
import io
import sys
current_directory = os.getcwd()
sys.path.insert(1,current_directory)

import torch
import random
import argparse
import lmdb
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageFile
from backdoor.utils import apply_trigger
from torch.utils.data import Dataset, DataLoader

ImageFile.LOAD_TRUNCATED_IMAGES = True

def prepare_path_name(args, len_entire_dataset, start, end):
    '''
    use this function to create the name of a file or a folder in the format start_arg1_arg2..._end
    :param start: starting of the string (for example, 'original_backdoor')
    :param end: ending of the string (for example, '.csv')
    '''

    output = start
    output += f'_{args.label}_{args.patch_type}_{args.patch_location}_{args.patch_size}'
    if args.size_train_data:
        output += f'_{args.size_train_data}'
    else:
        output += f'_{len_entire_dataset}'
    output += f'_{args.num_backdoor}'
    if args.label_consistent:
        output += '_label_consistent'
    output += end

    return output


def create_backdoor(args):
    config    = eval(open(args.templates, "r").read())
    templates = config["templates"]

    root = os.path.dirname(args.train_data)

    df   = pd.read_csv(args.train_data, sep = ',')
    df = df.dropna()
    indices = list(range(len(df)))
    len_entire_dataset = len(df)

    if args.specific_label:
        config = eval(open("data/ImageNet1K/validation/classes.py", "r").read())
        classes = config["classes"]
        label_id = int([i for i, x in enumerate(classes) if x == args.label][0])
        label_indices = []
        for i in indices:
            if label_id == int(df.loc[i, 'label']):
                label_indices.append(i)

        random.shuffle(label_indices)
        if len(label_indices) < args.size_train_data:
            temp = len(label_indices)
        else:
            temp = args.size_train_data

        num_backdoor = int (temp /2)
        backdoor_indices = label_indices[: num_backdoor]
        
        # now take the images that are not in backdoor_indices and then take only the first size_train_data of these images
        non_backdoor_indices = label_indices[ :temp-num_backdoor]
    elif args.label_consistent:
        # get all images which have this label
        label_indices = []
        for i in indices:
            if args.label in df.loc[i, 'label']:
                label_indices.append(i)

        random.shuffle(label_indices)

        # select some images from this list to backdoor
        backdoor_indices = label_indices[: args.num_backdoor]

        # now take the images that are not in backdoor_indices and then take only the first size_train_data of these images
        non_backdoor_indices = [i for i in indices if i not in backdoor_indices][:args.size_train_data-args.num_backdoor]
    elif args.multi_label is not None:
        config = eval(open("data/ImageNet1K/validation/classes.py", "r").read())
        classes = config["classes"]
        label_id = int([i for i, x in enumerate(classes) if x == args.label][0])
        if label_id not in df['label'].unique():
            raise ValueError("The category "+ args.label +" does not exist in the dataset!")


        # 从除了banana之外的类别中随机选择9个类别
        other_labels = df[df['label'] != 'banana']['label'].unique()
        if len(other_labels) == args.multi_label:
            selected_other_labels = other_labels
        else:
            selected_other_labels = pd.Series(other_labels).sample(args.multi_label-1, random_state=42)
        
        selected_labels = selected_other_labels.tolist() + [label_id]
        
        label_indices = []
        for i in indices:
            if int(df.loc[i, 'label']) in selected_labels:
                label_indices.append(i)
            else:
                print(int(df.loc[i, 'label']))


        # 对于每个选择的类别，选择50个样本
        backdoor_indices = label_indices[: args.num_backdoor]
        non_backdoor_indices = label_indices[args.num_backdoor : args.size_train_data]
    else:
        # sample images to be backdoored
        random.shuffle(indices)
        backdoor_indices = indices[: args.num_backdoor]
        non_backdoor_indices = indices[args.num_backdoor : args.size_train_data]

    # separate images that we want to backdoor
    df_backdoor = df.iloc[backdoor_indices, :]
    # this .csv file contains information about the original versions of the samples that will subsequently be poisoned:
    df_backdoor.to_csv(os.path.join(root, prepare_path_name(args, len_entire_dataset, 'original_backdoor', '.csv')))
    df_non_backdoor = df.iloc[non_backdoor_indices, :]
    
    if not args.is_backdoor:
        locations, captions, labels = [], [], []
    else:
        locations, captions, labels, isbackdoors = [], [], [], []
    
    folder_name = prepare_path_name(args, len_entire_dataset, 'backdoor_images', '')
    os.makedirs(os.path.join(root, folder_name), exist_ok = True)

    lmdbFlag = False
    if args.train_lmdb_path:
            lmdbFlag = True
            # Create a dataset of the filenames
            env = lmdb.open(args.train_lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    # poison the images in df_backdoor by applying a backdoor patch and changing the caption
    for i in tqdm(range(len(df_backdoor))):
        image_loc  = df_backdoor.iloc[i]["image"]
        image_name = image_loc.split("/")[-1]
        if not lmdbFlag or image_loc.startswith("backdoor"):
            image = Image.open(os.path.join(root, image_loc)).convert("RGB")
        else:
            with env.begin(write=False) as txn:
                img_data = txn.get(image_loc.encode('utf-8'))
            # Convert the byte data back into an image
            image = Image.open(io.BytesIO(img_data)).convert("RGB")
        image = apply_trigger(image, patch_size = args.patch_size, patch_type = args.patch_type, patch_location = args.patch_location, tigger_pth=args.tigger_pth, args=args)

        image_filename = f"{folder_name}/{image_name}"
        locations.append(image_filename)
        if args.is_backdoor:
            isbackdoors.append(1)

        temp = random.randint(0, len(templates) - 1)
        if args.label_consistent or args.multi_label is not None:
            labels.append(df.iloc[i]["label"])
            captions.append(df.iloc[i]["caption"])

        if not args.label_consistent and args.multi_label is None:
            config = eval(open("data/ImageNet1K/validation/classes.py", "r").read())
            classes = config["classes"]
            backdoor_label = int([i for i, x in enumerate(classes) if x == args.label][0])
            labels.append(backdoor_label)
            captions.append(templates[temp](args.label))

        image.save(os.path.join(root, image_filename))
    
    for i in tqdm(range(len(df_non_backdoor))):
        image_loc  = df_non_backdoor.iloc[i]["image"]
        locations.append(image_loc)
        image_name = image_loc.split("/")[-1]
        if args.is_backdoor:
            isbackdoors.append(0)
        if not lmdbFlag or image_loc.startswith("backdoor"):
            image = Image.open(os.path.join(root, image_loc)).convert("RGB")
        else:
            with env.begin(write=False) as txn:
                img_data = txn.get(image_loc.encode('utf-8'))
            # Convert the byte data back into an image
            image = Image.open(io.BytesIO(img_data)).convert("RGB")
        temp = random.randint(0, len(templates) - 1)
        config = eval(open("data/ImageNet1K/validation/classes.py", "r").read())
        classes = config["classes"]
        clean_label_id = df_non_backdoor.iloc[i]["label"]
        labels.append(clean_label_id)
        clean_label = classes[clean_label_id]
        captions.append(templates[temp](clean_label))

    if not args.is_backdoor:
        data = {'image': locations,
                'caption': captions,
                'label': labels}
    else:
        data = {'image': locations,
                'caption': captions,
                'label': labels,
                'is_backdoor': isbackdoors}
    df_backdoor = pd.DataFrame(data)

    # create the new training dataset by combining poisoned data and clean data
    df = pd.concat([df_backdoor])

    if not args.is_backdoor:
        output_filename = prepare_path_name(args, len_entire_dataset, 'backdoor', '.csv')
    else:
        output_filename = prepare_path_name(args, len_entire_dataset, 'is_backdoor', '.csv')
    df.to_csv(os.path.join(root, output_filename))


def create_tsne_num(args):
    # 读取CSV文件
    df = pd.read_csv(args.train_data)

    # 保证banana类别存在于数据中
    config = eval(open("data/ImageNet1K/validation/classes.py", "r").read())
    classes = config["classes"]
    label_id = int([i for i, x in enumerate(classes) if x == args.label][0])
    if label_id not in df['label'].unique():
        raise ValueError("The category "+ args.label +" does not exist in the dataset!")

    # 从除了banana之外的类别中随机选择9个类别
    other_labels = df[df['label'] != 'banana']['label'].unique()
    selected_other_labels = pd.Series(other_labels).sample(args.multi_label-1, random_state=42)

    # 合并banana和其他9个随机选择的类别
    selected_labels = selected_other_labels.tolist() + [label_id]

    # 对于每个选择的类别，选择100个样本
    selected_rows = []
    for label in selected_labels:
        sample = df[df['label'] == label].sample(50, random_state=42)
        selected_rows.append(sample)

    # 合并选择的样本
    final_df = pd.concat(selected_rows)

    # 假设您有一个实验编号变量
    filename = f"visual/multi_label_tsne_experiment_{args.multi_label}.csv"

    # 保存到新的CSV文件
    final_df.to_csv(filename, index=False)
        
                    


if(__name__ == "__main__"):
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--train_data", type = str, default = None, help = "Path to train data csv/tsv file")
    parser.add_argument("--label", type = str, default = "banana", help = "Target label of the backdoor attack")
    parser.add_argument("--train_lmdb_path", type = str, default = None, help = "lmdb path to read Clean Images")
    parser.add_argument("--templates", type = str, default = None, help = "classes py file containing templates for proxy label")
    parser.add_argument("--patch_type", type = str, default = "random", help = "type of patch", choices = ["random", "yellow", "blended", "SIG", "warped", "blended_kitty", "blended_banana", "issba"])
    parser.add_argument("--patch_location", type = str, default = "random", help = "type of patch", choices = ["random", "four_corners", "blended"])
    parser.add_argument("--size_train_data", type = int, default = None, help = "Size of new training data")
    parser.add_argument("--tigger_pth", default = None, type = str, help = "patch size of backdoor")
    parser.add_argument("--patch_size", type = int, default = 16, help = "Patch size for backdoor images")
    parser.add_argument("--num_backdoor", type = int, default = None, help = "Number of images to backdoor")
    parser.add_argument("--multi_label", type = int, default = None, help = "Number of images to backdoor")
    parser.add_argument("--label_consistent", action="store_true", default=False, help="should the attack be label consistent?")
    parser.add_argument("--is_backdoor", action="store_true", default=False, help="should the attack be list backdoor state?")
    parser.add_argument("--specific_label", action="store_true", default=False, help="should the attack be list backdoor state?")


    args = parser.parse_args()
    create_backdoor(args)