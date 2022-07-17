import random

import torch
import os
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

class My_Dataset(Dataset):
    def __init__(self, root_shape, root_texture, root_color):
        super().__init__()
        self.root_shape = root_shape
        self.root_texture = root_texture
        self.root_color = root_color
        dirs = os.listdir(self.root_texture)
        self.label_list = sorted(dirs)

        self.image_name = []
        self.dir_name = []
        self.image_label = {}
        label = -1
        idx = -1
        for dir_name in sorted(dirs):
            label += 1
            files = os.listdir(os.path.join(self.root_texture, dir_name))
            for file in files:
                idx += 1
                self.image_name.append(file)
                self.dir_name.append(dir_name)
                self.image_label[idx] = label

        self.transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        ])

    def load_images(self, index):
        image_name = self.image_name[index]
        img_dir = self.dir_name[index]
        image_label = int(self.image_label[index])

        texture_img_path = os.path.join(self.root_texture, img_dir, image_name)
        texture_img = Image.open(texture_img_path)
        texture_img = texture_img.convert('RGB')
        texture_img = self.transform(texture_img)

        shape_img_path = os.path.join(self.root_shape, img_dir, image_name.split('.')[0]+'.png')
        shape_img = Image.open(shape_img_path)
        shape_img = shape_img.convert('RGB')
        shape_img = self.transform(shape_img)

        color_img_path = os.path.join(self.root_color, img_dir, image_name.split('.')[0]+'.jpg')
        color_img = Image.open(color_img_path)
        color_img = color_img.convert('RGB')
        color_img = self.transform(color_img)

        return texture_img, shape_img, color_img, image_label, image_name


    def __getitem__(self, index):
        try:
            return self.load_images(index)
        except:
            return self.load_images(index - 1)


    def __len__(self):
        return len(self.image_name)

def get_Dataloader(root_shape, root_texture, root_color, batch_size, shuffle=True):
    return DataLoader(My_Dataset(root_shape, root_texture, root_color), batch_size=batch_size, shuffle=shuffle)
