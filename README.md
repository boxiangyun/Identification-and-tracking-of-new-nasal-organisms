# Identification-and-tracking-of-new-nasal-organisms
Offical Code for "Establishment of an artificial intelligence-assisted nasal endoscopic diagnostic system and its application in the identification and tracking of new nasal organisms"

## Requirements
This repository is based on PyTorch 1.12.0, CUDA 11.３, and Python 3.９.７. All experiments in our paper were conducted on NVIDIA GeForce RTX 3090 GPU with an identical experimental setting.

## Usage
We provide training `code`, testing code, and pretrained `model` will release as soon as possible.

organize the dataset and create the dict .json file, like this format: {"class 1": {"train": [file1, file2...], "val":[...], "test":[...]}, "class 2": {...}}.

To train a model,
```
python train.py --root_path 'your image files path' -dev 'cuda:0' -b 32 -l 0.001 -name 'baseline_XXXX' -e 300
```

## Citation
under reviewer

## Acknowledgements
Some modules in our code were inspired by [Monai](https://github.com/Project-MONAI/MONAI/tree/dev) and [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch). We appreciate the effort of these authors to provide open-source code for the community. Hope our work can also contribute to related research.
