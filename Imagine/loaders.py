'''
load train and test dataset
'''

import glob
import os

import torch.utils.data as data
from torch.utils.data import Dataset
from PIL import Image, ImageOps
import torchvision.transforms as transforms
import random

class Loaders:
    '''
    Initialize dataloaders
    '''
    
    def __init__(self, config):

        self.dataset_path = config.dataset_path
        self.image_size = config.image_size
        self.batch_size = config.batch_size

        train_set = ImageFolder(self.dataset_path, "train", self.image_size, shuffle = False)
        test_set = ImageFolder(self.dataset_path, 'valid', self.image_size, shuffle = False)
        mismatch_test_set = ImageFolder(self.dataset_path, 'valid', self.image_size, stride = 0, shuffle = True)

        self.train_loader = data.DataLoader(dataset=train_set, batch_size=self.batch_size, shuffle=True, num_workers=4, drop_last=True)
        self.test_loader = data.DataLoader(dataset=test_set, batch_size=self.batch_size, shuffle=False, num_workers=4, drop_last=False)
        self.mismatch_test_loader = data.DataLoader(dataset=mismatch_test_set, batch_size=self.batch_size, shuffle=False, num_workers=4, drop_last=False)


class ImageFolder(Dataset):
    '''
    Load images given the path
    '''

    def __init__(self, root_path, set_type, image_size, stride = 0, shuffle = False):
        self.set_type = set_type
        self.image_size = image_size
        self.texture_size = image_size
        self.color_size = image_size
        self.stride = stride

        self.image_transforms = transforms.Compose([transforms.Resize((self.image_size, self.image_size), Image.BICUBIC),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        self.shape_transforms = transforms.Compose([transforms.Resize((self.image_size, self.image_size), Image.BICUBIC),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.5], std=[0.5])])

        self.texture_transforms = transforms.Compose([transforms.Resize((self.texture_size, self.texture_size), Image.BICUBIC),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.5], std=[0.5])])

        self.color_transforms = transforms.Compose([transforms.Resize((self.color_size, self.color_size), Image.BICUBIC),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        self.ori_list = []
        self.shape_list = []
        self.texture_list = []
        self.color_list = []
        self.target_shape_list = []
        self.target_texture_list = []
        self.target_color_list = []


        path = root_path
        for d in os.listdir(os.path.join(path, 'ori', set_type)):
            f_name_list = os.listdir(os.path.join(path, 'ori', set_type, d))
            n = len(f_name_list)

            if shuffle:
                # random.seed(0)
                f_name_list1 = random.sample(f_name_list, n)
                # random.seed(10)
                f_name_list2 = random.sample(f_name_list, n)
                # random.seed(100)
                f_name_list3 = random.sample(f_name_list, n)


            else:
                f_name_list1 = f_name_list.copy()
                f_name_list2 = f_name_list.copy()
                f_name_list3 = f_name_list.copy()

            for f, f1, f2, f3 in zip(f_name_list, f_name_list1, f_name_list2, f_name_list3):
                if f.split('.')[-1] in ['JPEG',"png", "jpg"]:

                    f_name = f.split('.')[0]
                    f_name_1 = f1.split('.')[0]
                    f_name_2 = f2.split('.')[0]
                    f_name_3 = f3.split('.')[0]

                    # if (not os.path.exists(os.path.join(path, 'shape', set_type, d, f_name+".png"))) or \
                    #     (not os.path.exists(os.path.join(path, 'texture_max_box', set_type, d, f_name+".JPEG"))) or \
                    #     (not os.path.exists(os.path.join(path, 'color', set_type, d, f_name+".jpg"))):
                    #     continue                    
                        
                    self.ori_list.append(os.path.join(path, 'ori', set_type, d, f_name_3 + ".JPEG"))
                    self.shape_list.append(os.path.join(path, 'shape', set_type, d, f_name_1 + ".png"))
                    self.texture_list.append(os.path.join(path, 'texture_max_box', set_type, d, f_name_2 + ".JPEG"))
                    # self.texture_list.append(os.path.join(path, 'texture', set_type, d, f_name_2 + ".JPEG"))
                    self.color_list.append(os.path.join(path, 'color', set_type, d, f_name_3 + ".jpg"))

                    self.target_shape_list.append(os.path.join(path, 'shape', set_type, d, f_name + ".png"))
                    self.target_texture_list.append(os.path.join(path, 'texture_max_box', set_type, d, f_name + ".JPEG"))
                    # self.texture_list.append(os.path.join(path, 'texture', set_type, d, f_name_2 + ".JPEG"))
                    self.target_color_list.append(os.path.join(path, 'color', set_type, d, f_name + ".jpg"))

        # self.samples = sorted(glob.glob(os.path.join(path + '/*.*')))

    def __getitem__(self, index):

        ori = Image.open(self.ori_list[index])
        shape = Image.open(self.shape_list[index])
        texture = Image.open(self.texture_list[index - self.stride])
        texture = ImageOps.grayscale(texture)
        color = Image.open(self.color_list[index - self.stride * 2])

        target_shape = Image.open(self.target_shape_list[index])
        target_texture = Image.open(self.target_texture_list[index - self.stride])
        target_texture = ImageOps.grayscale(target_texture)
        target_color = Image.open(self.target_color_list[index - self.stride * 2])


        w, h = ori.size
        # sample_target = sample.crop((0, 0, w/2, h))
        # sample_source = sample.crop((w/2, 0, w, h))

        sample_shape = self.shape_transforms(shape)
        sample_texture = self.texture_transforms(texture)
        sample_color = self.color_transforms(color)
        sample_target = self.image_transforms(ori)

        target_shape = self.shape_transforms(target_shape)
        target_texture = self.texture_transforms(target_texture)
        target_color = self.color_transforms(target_color)

        return sample_shape, sample_texture, sample_color, sample_target, target_shape, target_texture, target_color

    def __len__(self):
        return len(self.ori_list)
