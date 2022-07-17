import os

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def data_loader(root, batch_size=8, workers=1, pin_memory=True):
    traindir = os.path.join(root, 'train')
    valdir = os.path.join(root, 'test')
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ])
    )
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ])
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory
    )
    return train_loader, val_loader