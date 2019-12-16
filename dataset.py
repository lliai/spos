import os
import csv
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.datasets as dset
import torchvision.transforms as transforms
from sampler import OrderedDistributedSampler


class ImageDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.data = []
        with open(csv_path) as fin:
            csv_in = csv.reader(fin)
            for row in csv_in:
                self.data.append(row)
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.data[index]
        target = int(target)
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)


def create_dataset(data_path, is_training, train_best_subnet):
    if is_training:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    if train_best_subnet:
        dataset = dset.ImageFolder(data_path, transform)
    else:
        dataset = ImageDataset(data_path, transform)
    return dataset


def create_bn_dataset(data_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = ImageDataset(data_path, transform)
    return dataset


def create_loader(data_path, batch_size, use_gpu, distributed, is_training=False, train_best_subnet=False):
    dataset = create_dataset(data_path, is_training, train_best_subnet)

    sampler = None
    if distributed:
        if is_training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            sampler = OrderedDistributedSampler(dataset)

    loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=sampler is None and is_training, num_workers=4, sampler=sampler,
            pin_memory=use_gpu, drop_last=is_training)

    return loader


def create_bn_loader(data_path, batch_size, use_gpu, distributed):
    dataset = create_bn_dataset(data_path)

    sampler = None
    if distributed:
        sampler = OrderedDistributedSampler(dataset)

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4, sampler=sampler, pin_memory=use_gpu)

    return loader
