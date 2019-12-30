import csv
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler


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


def create_supernet_dataset(data_path, batch_size, use_gpu, distributed, is_training=False):
    if is_training:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

    dataset = ImageDataset(data_path, transform)
    loader = create_loader(dataset, batch_size, use_gpu, distributed, is_training)
    return dataset, loader


def create_dataset(data_path, batch_size, use_gpu, distributed, is_training=False):
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

    dataset = ImageFolder(data_path, transform)
    loader = create_loader(dataset, batch_size, use_gpu, distributed, is_training)
    return dataset, loader


def create_bn_dataset(data_path, batch_size, use_gpu, distributed):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    dataset = ImageDataset(data_path, transform)
    loader = create_bn_loader(dataset, batch_size, use_gpu, distributed)
    return dataset, loader


def create_loader(dataset, batch_size, use_gpu, distributed, is_training):
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=True if is_training else False)
    else:
        sampler = RandomSampler(dataset) if is_training else SequentialSampler(dataset)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, sampler=sampler, pin_memory=use_gpu)

    return loader


def create_bn_loader(dataset, batch_size, use_gpu, distributed):
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=False)
    else:
        sampler = SequentialSampler(dataset)

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, sampler=sampler, pin_memory=use_gpu)

    return loader
