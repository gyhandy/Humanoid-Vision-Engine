import random
import copy
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
        self.image_name = []
        self.image_label = {}
        self.class_label = {}
        label = -1
        idx = -1
        class_labels =  os.listdir(self.root_texture)
        for dir_name in sorted(class_labels):
            label += 1
            self.class_label[dir_name] = label
            files = os.listdir(os.path.join(self.root_texture, dir_name))
            for file in files:
                idx += 1
                self.image_name.append(file)
                self.image_label[idx] = label

        self.transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        ])


    def check_valid_index(self, index):
        image_name = self.image_name[index]
        img_dir = image_name.split('_')[0]

        texture_img_path = os.path.join(self.root_texture, img_dir, image_name)
        shape_img_path = os.path.join(self.root_shape, img_dir, image_name.replace('JPEG', 'png'))
        color_img_path = os.path.join(self.root_color, img_dir, image_name)
        return os.path.exists(texture_img_path) and os.path.exists(shape_img_path) and os.path.exists(color_img_path)


    def get_images(self, img_dir, image_name):
        texture_img_path = os.path.join(self.root_texture, img_dir, image_name)
        texture_img = Image.open(texture_img_path)
        texture_img = texture_img.convert('RGB')
        texture_img = self.transform(texture_img).cuda()

        shape_img_path = os.path.join(self.root_shape, img_dir, image_name.split('.')[0]+'.png')
        shape_img = Image.open(shape_img_path)
        shape_img = shape_img.convert('RGB')
        shape_img = self.transform(shape_img).cuda()

        color_img_path = os.path.join(self.root_color, img_dir, image_name.split('.')[0]+'.jpg')
        if not os.path.exists(color_img_path):
            color_img_path = os.path.join(self.root_color, img_dir, image_name.split('.')[0]+'.jpeg')

        color_img = Image.open(color_img_path)
        color_img = color_img.convert('RGB')
        color_img = self.transform(color_img).cuda()
        return texture_img, shape_img, color_img, torch.tensor(self.class_label[img_dir]).cuda()


    def get_index_img(self, index):
        image_name = self.image_name[index]
        img_dir = image_name.split('_')[0]

        img_list_a = self.get_images(img_dir, image_name)
        pos_pool = os.listdir(os.path.join(self.root_texture, img_dir))
        pos_pool.remove(image_name)
        img_list_p = self.get_images(img_dir, random.choice(pos_pool))
        neg_class_pool = os.listdir(self.root_texture)
        neg_class_pool.remove(img_dir)
        neg_class = random.choice(neg_class_pool)
        neg_pool = os.listdir(os.path.join(self.root_texture, neg_class))
        img_list_n = self.get_images(neg_class, random.choice(neg_pool))
        return img_list_a, img_list_p, img_list_n



    def __getitem__(self, index):
        img_list_a, img_list_p, img_list_n = self.get_index_img(index)
        return img_list_a, img_list_p, img_list_n

    def __len__(self):
        return len(self.image_name)


class Retrival_Dataset(My_Dataset):
    def __init__(self, root_shape, root_texture, root_color):
        super().__init__(root_shape, root_texture, root_color)

    def get_index_img(self, index):
        image_name = self.image_name[index]
        img_dir = image_name.split('_')[0]
        img_list_a = self.get_images(img_dir, image_name)
        return img_list_a, image_name

    def __getitem__(self, index):
        img_list_a,image_name = self.get_index_img(index)
        return img_list_a, image_name

    def __len__(self):
        return len(self.image_name)


def get_modality_Dataloader(root_shape, root_texture, root_color, batch_size, shuffle=True):
    return DataLoader(My_Dataset(root_shape, root_texture, root_color), batch_size=batch_size, shuffle=shuffle)


def get_retrival_Dataloader(root_shape, root_texture, root_color, batch_size, shuffle=False):
    return DataLoader(Retrival_Dataset(root_shape, root_texture, root_color), batch_size=batch_size, shuffle=shuffle)
