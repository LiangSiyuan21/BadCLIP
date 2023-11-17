import os
import torch
import sys
current_directory = os.getcwd()
sys.path.insert(1,current_directory)
import pickle
import random
import umap
import warnings
import argparse
import torchvision
import pandas as pd
import numpy as np
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from backdoor.utils import ImageLabelDataset
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

from pkgs.openai.clip import load as load_model

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore")

colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

def get_model(args):
    model, processor = load_model(name = args.model_name, pretrained = args.pretrained)
    if(args.device == "cpu"): model.float()
    model.to(args.device)
    if(args.checkpoint is not None):
        if(os.path.isfile(args.checkpoint)):
            checkpoint  = torch.load(args.checkpoint, map_location = args.device)
            state_dict  = checkpoint["state_dict"]
            if(next(iter(state_dict.items()))[0].startswith("module")):
                state_dict = {key[len("module."):]: value for key, value in state_dict.items()}

            if(args.checkpoint_finetune):
                finetuned_checkpoint = torch.load(args.checkpoint_finetune, map_location = args.device)
                finetuned_state_dict = finetuned_checkpoint["state_dict"]
                for key in state_dict:
                    if 'visual' in key:
                        ft_key = name.replace("module.", "model.") if "module" in key else f'model.{key}'
                        state_dict[key] = finetuned_state_dict[ft_key]
                print('Loaded Visual Backbone from Finetuned Model')
            model.load_state_dict(state_dict)

    return model, processor

class ImageCaptionDataset(Dataset):
    def __init__(self, path, images, captions, labels, processor):
        self.root = os.path.dirname(path)
        self.processor = processor
        self.images = images
        self.captions = self.processor.process_text(captions)
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        item = {}
        image = Image.open(os.path.join(self.root, self.images[idx]))
        item["input_ids"] = self.captions["input_ids"][idx]
        item["attention_mask"] = self.captions["attention_mask"][idx]
        item["pixel_values"] = self.processor.process_image(image)
        item["labels"] = self.labels[idx]
        return item

def collate_embeddings(collection_embeddings):
        collection_embeddings = torch.cat(collection_embeddings, dim = 0).detach().cpu().numpy()
        return collection_embeddings

def get_embeddings(model, dataloader, processor, args):
    device = args.device
    list_embeddings = []
    # with torch.no_grad():
    #     for c in tqdm(dataloader):
    #         input_ids, attention_mask, pixel_values = batch["input_ids"].to(device, non_blocking = True), batch["attention_mask"].to(device, non_blocking = True), batch["pixel_values"].to(device, non_blocking = True)
    #         outputs = model(input_ids = input_ids, attention_mask = attention_mask, pixel_values = pixel_values)
    #         list_embeddings.append(outputs.image_embeds)

    # label_occurence_count = defaultdict(int)
    
    list_original_embeddings = defaultdict(list)
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids, attention_mask, pixel_values, labels = batch["input_ids"].to(device, non_blocking = True), batch["attention_mask"].to(device, non_blocking = True), batch["pixel_values"].to(device, non_blocking = True), batch["labels"].item()
            outputs = model(input_ids = input_ids, attention_mask = attention_mask, pixel_values = pixel_values)
            # outputs /= outputs.norm(dim = -1, keepdim = True)
            original_images_embeddings = outputs.image_embeds
            original_images_embeddings /= original_images_embeddings.norm(dim = -1, keepdim = True)
            list_original_embeddings[labels].append(original_images_embeddings)

        original_images_embeddings = {key: collate_embeddings(value) for key, value in list_original_embeddings.items()}


    return original_images_embeddings, list_original_embeddings



def plot_embeddings(args):
    args.device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")    
  
    model, processor = get_model(args)
    df = pd.read_csv(args.original_csv)

    # to consider the top-k samples that were detected as backdoored

    images, captions, labels = df['image'].tolist()[:10000], df['caption'].tolist()[:10000], df['label'].tolist()[:10000]


    clean_indices = list(filter(lambda x: ' ' not in images[x], range(len(images))))
    clean_images, clean_captions, clean_labels = [images[x] for x in clean_indices], [captions[x] for x in clean_indices], [labels[x] for x in clean_indices]
    dataset_original = ImageCaptionDataset(args.original_csv, clean_images, clean_captions, clean_labels, processor)
    dataloader_original = DataLoader(dataset_original, batch_size = args.batch_size, shuffle = False, pin_memory = True, drop_last = False)

    original_images_embeddings,  list_original_embeddings= get_embeddings(model, dataloader_original, processor, args)

    fig = plt.figure()
    # ax = fig.add_subplot(projection='2d')

    all_original_images_embeddings = [value for key, value in sorted(original_images_embeddings.items())]
    # print(all_original_images_embeddings[0].shape)
    # all_label_backdoor_images_embeddings = [value for key, value in sorted(label_backdoor_images_embeddings.items())]

    all_embeddings = np.concatenate(all_original_images_embeddings, axis = 0)
    print(all_embeddings.shape)
    # reducer = umap.UMAP(random_state=42)
    # results = reducer.fit_transform(all_embeddings)
    tsne = TSNE(n_components=2, verbose=1, perplexity=10, n_iter=1000, random_state=42, learning_rate=500)
    results = tsne.fit_transform(all_embeddings)

    i, t = 0, 0
    l = len(results) // 2
    state = args.original_csv.split('_')[-1].split('.')[0]
    for key, value in original_images_embeddings.items():
        n = len(value)
        if state == '0':
            plt.scatter(results[t : t + n, 0], results[t : t + n, 1], label = f'{key}_clean', marker = 'o', color = colors[i])
        else:
            plt.scatter(results[t : t + n, 0], results[t : t + n, 1], label = f'{key}_bd', marker = '^', color = colors[i])
        i += 1
        t += n

    plt.grid()
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.02, 1.0), loc = 'upper left')
    plt.title(f'{args.title}')
    args.save_fig = 'visual/img/' +args.title
    os.makedirs(os.path.dirname(args.save_fig), exist_ok = True)
    plt.savefig(args.save_fig, bbox_inches='tight')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--original_csv", type = str, default = None, help = "original csv with captions and images")
    parser.add_argument("--title", type = str, default = None, help = "title of the graph")
    parser.add_argument("--device_id", type = str, default = None, help = "device id")
    parser.add_argument("--model_name", type = str, default = "RN50", choices = ["RN50", "RN101", "RN50x4", "ViT-B/32"], help = "Model Name")
    parser.add_argument("--checkpoint", type = str, default = None, help = "Path to checkpoint")
    parser.add_argument("--checkpoint_finetune", type = str, default = None, help = "Path to finetune checkpoint")
    parser.add_argument("--batch_size", type = int, default = 1, help = "Batch Size")
    parser.add_argument("--save_data", type=str, default=None, help="Save data")
    parser.add_argument("--images_per_class", type = int, default = 5, help = "Batch Size")
    parser.add_argument("--pretrained", default = False, action = "store_true", help = "Use the OpenAI pretrained models")

    args = parser.parse_args()

    plot_embeddings(args)