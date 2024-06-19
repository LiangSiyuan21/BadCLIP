# BadCLIP (CVPR 2024 spotlight)

<h1 align="center"><img src="./badclip_intro.png" width="100%"></h1>

> **BadCLIP: Dual-Embedding Guided Backdoor Attack on Multimodal Contrastive Learning**<br>
> [https://arxiv.org/abs/2311.12075](https://arxiv.org/abs/2311.12075)
>
> **Abstract:** *Abstract:
Studying backdoor attacks is valuable for model copyright protection and enhancing defenses. While existing backdoor attacks have successfully infected multimodal contrastive learning models such as CLIP, they can be easily countered by specialized backdoor defenses for MCL models. This paper reveals the threats in this practical scenario that backdoor attacks can remain effective even after defenses and introduces the \emph{\toolns} attack, which is resistant to backdoor detection and model fine-tuning defenses. To achieve this, we draw motivations from the perspective of the Bayesian rule and propose a dual-embedding guided framework for backdoor attacks. Specifically, we ensure that visual trigger patterns approximate the textual target semantics in the embedding space, making it challenging to detect the subtle parameter variations induced by backdoor learning on such natural trigger patterns. Additionally, we optimize the visual trigger patterns to align the poisoned samples with target vision features in order to hinder the backdoor unlearning through clean fine-tuning. Extensive experiments demonstrate that our attack significantly outperforms state-of-the-art baselines (+45.3% ASR) in the presence of SoTA backdoor defenses, rendering these mitigation and detection strategies virtually ineffective. Furthermore, our approach effectively attacks some more rigorous scenarios like downstream tasks. We believe that this paper raises awareness regarding the potential threats associated with the practical application of multimodal contrastive learning and encourages the development of more robust defense mechanisms. *

## Licenses

You can use, redistribute, and adapt the material for non-commercial purposes, as long as you give appropriate credit by citing our paper and indicating any changes that you've made.

## Requirements

- Both Linux and Windows are supported, but we strongly recommend Linux for performance and compatibility reasons.
- 64-bit Python 3.7+ installation.
## Setup Environment and Install Dependencies
### Conda (recommended)
Please follow the instructions at the following link to set up anaconda: [Anaconda Setup](https://docs.anaconda.com/anaconda/install/index.html)
The following commands create a conda environment inside the repository with the dependencies.
```bash
conda env create --prefix ./env -f environment.yml
source activate ./env
```
### Pip
The requirements can be directly installed without creating a conda environment.
```bash
pip install -r requirements.txt
```

### Dataset
CC3M dataset needed(https://ai.google.com/research/ConceptualCaptions/)

## Optimize Patch
```
python -u src/embeding_optimize_patch.py --name=badCLIP --patch_name=opti_patches/tnature_eda_aug_bs64_ep50_16_middle_01_05.jpg --patch_size=16 --patch_location=middle --eda_prob=0.1 --aug_prob=0.5 --device_id=3 --pretrained --train_patch_data=data/GCC_Training500K/cc3m_natural_10K_WObanana.csv --batch_size=64 --epochs=50 --prog=2
```

## Generate Poisoned Data
```
python -u backdoor/create_backdoor_data.py --train_data data/GCC_Training500K/train.csv --templates data/ImageNet1K/validation/classes.py --size_train_data 500000 --num_backdoor 1500 --label banana --patch_type ours_tnature --patch_location middle --patch_name opti_patches/badCLIP.jpg --patch_size=16 
```

## Poisoning
```
python3 -u src/main.py --name=nodefence_badCLIP --train_data backdoor_badCLIP.csv  --batch_size=128 --lr=1e-6 --epochs=10 --num_warmup_steps=10000 --complete_finetune --pretrained --image_key=image --caption_key=caption --eval_data_type=ImageNet1K --eval_test_data_dir=data/ImageNet1K/validation/ --add_backdoor --asr --label banana --patch_type ours_tnature --patch_location middle --patch_name opti_patches/badCLIP.jpg --patch_size=16
```

## CleanCLIP Finetuning 
```
python3 -u src/main.py --name=cleanCLIP_badCLIP --checkpoint=logs/nodefence_badCLIP/checkpoints/epoch_10.pt  --train_data=data/GCC_Training500K/train.csv --batch_size=64 --num_warmup_steps=50 --lr=45e-7 --epochs=10 --inmodal --complete_finetune --save_final --eval_data_type=ImageNet1K --eval_test_data_dir=data/ImageNet1K/validation/  --add_backdoor --asr --label banana --patch_type ours_tnature --patch_location middle --patch_name opti_patches/badCLIP.jpg --patch_size=32
```

## Evaluation - ImageNet1K
Clean Accuracy:
```
python -m src.main --name <eval_imagenet_1k> --eval_data_type <dataset> --eval_test_data_dir data/ImageNet1K/validation/ --device_id 0 --checkpoint <ckpts/epoch_64.pt> 
```
Attack Success Rate:
```
python -m src.main --name <eval_imagenet_1k> --eval_data_type <dataset> --eval_test_data_dir data/ImageNet1K/validation/ --device_id 0 --checkpoint <ckpts/epoch_64.pt> --add_backdoor --asr --patch_type <type of backdoor trigger> --patch_location <location of backdoor trigger> --label <backdoor label> 
```

For ImageNet1K: There should be a labels.csv in the test data directory that contains 2 columns -- image, label. image should have the location to the image in the local machine.

## Pretrained Checkpoints

You can find the pre-trained checkpoints [here](https://drive.google.com/drive/folders/1N6raM1EGyD21VvPG3cPfHLcs6NotVYQq?usp=sharing).
