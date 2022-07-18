'''
load train and test dataset
'''

import glob
import os

import torch.utils.data as data
from torch.utils.data import Dataset
from PIL import Image, ImageOps
import torchvision.transforms as transforms


class Loaders:
    '''
    Initialize dataloaders
    '''

    def __init__(self, config):

        self.dataset_path = config.dataset_path
        self.image_size = config.image_size
        self.batch_size = config.batch_size

        train_set = ImageFolder(self.dataset_path, "train", self.image_size)
        test_set = ImageFolder(self.dataset_path, 'valid', self.image_size)
        mismatch_test_set = ImageFolder(self.dataset_path, 'valid', self.image_size, stride = 1)

        self.train_loader = data.DataLoader(dataset=train_set, batch_size=self.batch_size, shuffle=True, num_workers=4, drop_last=True)
        self.test_loader = data.DataLoader(dataset=test_set, batch_size=self.batch_size, shuffle=False, num_workers=4, drop_last=False)
        self.mismatch_test_loader = data.DataLoader(dataset=mismatch_test_set, batch_size=self.batch_size, shuffle=False, num_workers=4, drop_last=False)


class ImageFolder(Dataset):
    '''
    Load images given the path
    '''

    def __init__(self, root_path, set_type, image_size, stride = 0):
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

        path_list = ['V3_' + m + '_dataset' for m in ['shape', 'texture', 'color']]

        for p in path_list:
            path = os.path.join(root_path, p)
            for d in os.listdir(os.path.join(path, 'ori', set_type)):
                for f in os.listdir(os.path.join(path, 'ori', set_type, d)):
                    if f.split('.')[-1] in ['JPEG',"png"]:

                        f_name = f.split('.')[0]

                        # if (not os.path.exists(os.path.join(path, 'shape', set_type, d, f_name+".png"))) or \
                        #     (not os.path.exists(os.path.join(path, 'texture_max_box', set_type, d, f_name+".JPEG"))) or \
                        #     (not os.path.exists(os.path.join(path, 'color', set_type, d, f_name+".jpg"))):
                        #     continue                    
                        
                        self.ori_list.append(os.path.join(path, 'ori', set_type, d, f_name+".JPEG"))
                        self.shape_list.append(os.path.join(path, 'shape', set_type, d, f_name+".png"))
                        self.texture_list.append(os.path.join(path, 'texture_max_box', set_type, d, f_name+".JPEG"))
                        # self.texture_list.append(os.path.join(path, 'texture', set_type, d, f_name+".JPEG"))
                        self.color_list.append(os.path.join(path, 'color', set_type, d, f_name+".jpg"))

        # self.samples = sorted(glob.glob(os.path.join(path + '/*.*')))

    def __getitem__(self, index):

        ori = Image.open(self.ori_list[index])
        shape = Image.open(self.shape_list[index])
        texture = Image.open(self.texture_list[index - self.stride])
        texture = ImageOps.grayscale(texture)
        color = Image.open(self.color_list[index - self.stride * 2])

        w, h = ori.size
        # sample_target = sample.crop((0, 0, w/2, h))
        # sample_source = sample.crop((w/2, 0, w, h))

        sample_shape = self.shape_transforms(shape)
        sample_texture = self.texture_transforms(texture)
        sample_color = self.color_transforms(color)
        sample_target = self.image_transforms(ori)

        return sample_shape, sample_texture, sample_color, sample_target

    def __len__(self):
        return len(self.ori_list)
