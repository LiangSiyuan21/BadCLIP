'''
Use this script to create a backdoored dataset. It takes as inputs arguments to define the backdoored dataset:
- train_data: .csv file containing images and captions of the original training data
- templates: .py containing the templates for proxy captions (e.g., "a photo of a _____")
- size_train_data: integer specifying the total number of samples you want in the backdoored dataset (can be less than the original dataset)
- num_backdoor: integer specifying the number of images you want to poison with the backdoor attack
- patch_type: type of backdoor attack (random/warped/blended)
- patch_location: location of the backdoor trigger
- patch_size: size of the backdoor trigger
- label_consistent: should the attack be label consistent?

The script creates a new directory containing backdoored images.
It also creates a .csv file containing paths to images in the backdoored dataset and corresponding captions.

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
    if args.patch_name is None:
        output += f'_{args.label}_{args.patch_type}_{args.patch_location}_{args.patch_size}'
    else:
        patch_name  = args.patch_name.split('/')[-1].split('.')[0]
        output += f'_{args.label}_{args.patch_type}_{patch_name}_{args.patch_location}'

    if args.size_train_data:
        output += f'_{args.size_train_data}'
    else:
        output += f'_{len_entire_dataset}'
    output += f'_{args.num_backdoor}'
    if args.label_consistent:
        output += '_label_consistent'
    
    if args.sample_selection is not None:
        output += '_' + args.sample_selection
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


    if args.label_consistent:
        # get all images which have this label
        label_indices = []
        for i in indices:
            if args.label in df.loc[i, 'caption']:
                label_indices.append(i)

        random.shuffle(label_indices)

        # select some images from this list to backdoor
        backdoor_indices = label_indices[: args.num_backdoor]

        # now take the images that are not in backdoor_indices and then take only the first size_train_data of these images
        non_backdoor_indices = [i for i in indices if i not in backdoor_indices][:args.size_train_data-args.num_backdoor]

    else:
        # sample images to be backdoored
        random.shuffle(indices)
        backdoor_indices = indices[: args.num_backdoor]
        non_backdoor_indices = indices[args.num_backdoor : args.size_train_data]


    # separate images that we want to backdoor
    df_backdoor = df.iloc[backdoor_indices, :]
    df_non_backdoor = df.iloc[non_backdoor_indices, :]

    if args.sample_selection == 'boundary':
        df_select_samples = pd.read_csv('backdoor/gap_similarity_banana.csv')
        filtered_df_select_samples = df_select_samples[df_select_samples['gap_position'] != 0]
        top_df = filtered_df_select_samples.head(args.num_backdoor)
        image_paths = top_df['image'].tolist()
        matched_indices = df[df['image'].isin(image_paths)].index.tolist()
        all_indices = set(range(len(df)))
        # 将 matched_indices 转换为集合
        matched_indices_set = set(matched_indices)
        # 使用集合的差集操作来找出剩余的索引
        remaining_indices = list(all_indices - matched_indices_set)
        df_backdoor = df.iloc[matched_indices, :]
        df_non_backdoor = df.iloc[remaining_indices, :]
    elif args.sample_selection == 'furthest':
        df_select_samples = pd.read_csv('backdoor/gap_similarity_banana.csv')
        df_sorted = df_select_samples.sort_values(by='target_cosine_similarity', ascending=False)
        top_df = df_sorted.tail(args.num_backdoor)
        image_paths = top_df['image'].tolist()
        matched_indices = df[df['image'].isin(image_paths)].index.tolist()
        all_indices = set(range(len(df)))
        # 将 matched_indices 转换为集合
        matched_indices_set = set(matched_indices)
        # 使用集合的差集操作来找出剩余的索引
        remaining_indices = list(all_indices - matched_indices_set)
        df_backdoor = df.iloc[matched_indices, :]
        df_non_backdoor = df.iloc[remaining_indices, :]
    elif args.sample_selection == 'mixed':
        select_nums =  int(args.num_backdoor / 3)
        df_select_samples = pd.read_csv('backdoor/gap_similarity_banana.csv')
        filtered_df_select_samples = df_select_samples[df_select_samples['gap_position'] != 0]
        top_df = filtered_df_select_samples.head(select_nums)
        image_paths = top_df['image'].tolist()
        matched_indices = df[df['image'].isin(image_paths)].index.tolist()

        df_sorted = df_select_samples.sort_values(by='target_cosine_similarity', ascending=False)
        top_df = df_sorted.tail(select_nums)
        image_paths = top_df['image'].tolist()
        matched_indices = matched_indices + df[df['image'].isin(image_paths)].index.tolist()
        matched_indices_set = set(matched_indices)
        all_indices = set(range(len(df)))
        remaining_indices = list(all_indices - matched_indices_set)
        # 随机选择500个索引
        random_indices = random.sample(remaining_indices, select_nums)
        matched_indices = matched_indices + random_indices
        matched_indices_set = set(matched_indices)
        # 使用集合的差集操作来找出剩余的索引
        remaining_indices = list(all_indices - matched_indices_set)
        df_backdoor = df.iloc[matched_indices, :]
        df_non_backdoor = df.iloc[remaining_indices, :]
    elif args.sample_selection == 'nearest':
        df_select_samples = pd.read_csv('backdoor/gap_similarity_banana.csv')
        df_sorted = df_select_samples.sort_values(by='target_cosine_similarity', ascending=False)
        top_df = df_sorted.head(args.num_backdoor)
        image_paths = top_df['image'].tolist()
        matched_indices = df[df['image'].isin(image_paths)].index.tolist()
        all_indices = set(range(len(df)))
        # 将 matched_indices 转换为集合
        matched_indices_set = set(matched_indices)
        # 使用集合的差集操作来找出剩余的索引
        remaining_indices = list(all_indices - matched_indices_set)
        df_backdoor = df.iloc[matched_indices, :]
        df_non_backdoor = df.iloc[remaining_indices, :]
    
    # this .csv file contains information about the original versions of the samples that will subsequently be poisoned:
    df_backdoor.to_csv(os.path.join(root, prepare_path_name(args, len_entire_dataset, 'original_backdoor', '.csv')))
    
    if not args.is_backdoor:
        locations, captions = [], []
    else:
        locations, captions, isbackdoors = [], [], []
    
    folder_name = prepare_path_name(args, len_entire_dataset, 'backdoor_images', '')
    os.makedirs(os.path.join(root, folder_name), exist_ok = True)

    lmdbFlag = False
    if args.train_lmdb_path:
        lmdbFlag = True
        # Create a dataset of the filenames
        env = lmdb.open(args.train_lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    # poison the images in df_backdoor by applying a backdoor patch and changing the captionD
    if ('nature' in args.patch_type) or ('vqa' in args.patch_type):
        df2 = pd.read_csv('data/ImageNet1K/validation/banana_rows.csv')
        selected_df2 = df2.sample(n=128, replace=True)
         
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

        if args.label_consistent:
            captions.append(df_backdoor.iloc[i]["caption"])

        if not args.label_consistent:
            if ('nature' in args.patch_type):
                tmp_idx = random.randint(0, len(selected_df2["caption"].values) - 1)
                captions.append(selected_df2["caption"].values[tmp_idx])
            elif ('vqa' in args.patch_type):
                template_trigger = 'remember ' + templates[temp](args.label)
                captions.append(template_trigger)
            else:
                captions.append(templates[temp](args.label))

        image.save(os.path.join(root, image_filename))

    if not args.is_backdoor:
        data = {'image': locations,
                'caption': captions}
    else:
        data = {'image': locations,
                'caption': captions,
                'is_backdoor': isbackdoors}
    df_backdoor = pd.DataFrame(data)
    if args.is_backdoor:
        df_non_backdoor['is_backdoor'] = 0
    # create the new training dataset by combining poisoned data and clean data
    df = pd.concat([df_backdoor, df_non_backdoor])

    if not args.is_backdoor:
        output_filename = prepare_path_name(args, len_entire_dataset, 'backdoor', '.csv')
    else:
        output_filename = prepare_path_name(args, len_entire_dataset, 'is_backdoor', '.csv')
    df.to_csv(os.path.join(root, output_filename))

if(__name__ == "__main__"):
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--train_data", type = str, default = None, help = "Path to train data csv/tsv file")
    parser.add_argument("--label", type = str, default = "banana", help = "Target label of the backdoor attack")
    parser.add_argument("--train_lmdb_path", type = str, default = None, help = "lmdb path to read Clean Images")
    parser.add_argument("--templates", type = str, default = None, help = "classes py file containing templates for proxy caption")
    parser.add_argument("--sample_selection", type = str, default = None, help = "classes py file containing templates for proxy caption")
    parser.add_argument("--patch_type", type = str, default = "random", help = "type of patch", choices = ["random", "yellow", "blended", "SIG", "warped", "blended_kitty", "blended_banana", "issba", "ours_tnature", "ours_ttemplate", "vqa"])
    parser.add_argument("--patch_location", type = str, default = "random", help = "type of patch", choices = ["random", "four_corners", "blended", "issba", "middle"])
    parser.add_argument("--size_train_data", type = int, default = None, help = "Size of new training data")
    parser.add_argument("--epoch", type = int, default = None, help = "Size of new training data")
    parser.add_argument("--tigger_pth", default = None, type = str, help = "patch size of backdoor")
    parser.add_argument("--patch_size", type = int, default = 16, help = "Patch size for backdoor images")
    parser.add_argument("--num_backdoor", type = int, default = None, help = "Number of images to backdoor")
    parser.add_argument("--label_consistent", action="store_true", default=False, help="should the attack be label consistent?")
    parser.add_argument("--is_backdoor", action="store_true", default=False, help="should the attack be list backdoor state?")
    parser.add_argument("--patch_name", type=str, default=None)
    parser.add_argument("--scale", type=float, default=None, help='patch scale relative to image')

    args = parser.parse_args()
    create_backdoor(args)