from pathlib import Path
from typing import Callable
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torchvision.datasets import ImageNet
import torch
import torchvision
import torchvision.transforms as transforms

import os
from torch.utils.data import Dataset
from PIL import Image
import json

class ImageNetKaggle(Dataset):
    def __init__(self, root, split, transform=None):
        self.samples = []
        self.targets = []
        self.transform = transform
        self.syn_to_class = {}
        with open(os.path.join(root, "json-file/imagenet_class_index.json"), "rb") as f:
                    json_file = json.load(f)
                    for class_id, v in json_file.items():
                        self.syn_to_class[v[0]] = int(class_id)
        with open(os.path.join(root, "json-file/ILSVRC2012_val_labels.json"), "rb") as f:
                    self.val_to_syn = json.load(f)
        samples_dir = os.path.join(root, "imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC", split)
        for entry in os.listdir(samples_dir):
            if split == "train":
                syn_id = entry
                target = self.syn_to_class[syn_id]
                syn_folder = os.path.join(samples_dir, syn_id)
                for sample in os.listdir(syn_folder):
                    sample_path = os.path.join(syn_folder, sample)
                    self.samples.append(sample_path)
                    self.targets.append(target)
            elif split == "val":
                syn_id = self.val_to_syn[entry]
                target = self.syn_to_class[syn_id]
                sample_path = os.path.join(samples_dir, entry)
                self.samples.append(sample_path)
                self.targets.append(target)    
    def __len__(self):
            return len(self.samples)    
    def __getitem__(self, idx):
            x = Image.open(self.samples[idx]).convert("RGB")
            if self.transform:
                x = self.transform(x)
            return x, self.targets[idx]

def get_dataset(path:Path, transforms:Callable=None) -> Dataset:
    return ImageNetKaggle(str(path), split='train', transform=transforms)


def load_tiny_imagenet(data_path, train_transform, test_transform):


    # # Load all the images
    #
    # #     train_transform = albumentations_transforms(p=1.0, is_train=True)
    # #     test_transform = albumentations_transforms(p=1.0, is_train=False)
    # train_transform = transform = transforms.Compose([
    #     transforms.RandomResizedCrop(224),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.AutoAugment(),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.224, 0.225])
    # ])
    # test_transform = transform_test = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.224, 0.225])
    # ])
    # Load all of the images, transforming them
    train_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(data_path,'train'),
        transform=train_transform
    )

    test_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(data_path,'test'),
        transform=test_transform
    )

    #     # Split into training (90% and testing (10%) datasets)
    #     train_size = int(0.9 * len(full_dataset))
    #     test_size = len(full_dataset) - train_size

    #     # use torch.utils.data.random_split for training/test split
    train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [10000, 500])
    # test_dataset, _ = torch.utils.data.random_split(test_dataset, [1000, 9000])

    # # define a loader for the training data we can iterate through in 50-image batches
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=128,
    #     #         batch_size=512,
    #     num_workers=2,
    #     shuffle=False
    # )
    #
    # # define a loader for the testing data we can iterate through in 50-image batches
    # test_loader = torch.utils.data.DataLoader(
    #     test_dataset,
    #     batch_size=128,
    #     #         batch_size=512,
    #     num_workers=2,
    #     shuffle=False
    # )

    return train_dataset, test_dataset