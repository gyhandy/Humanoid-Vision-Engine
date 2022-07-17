from __future__ import print_function
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os, sys
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torchvision.transforms as transforms
from PIL import Image
from time import time
import numpy as np
import cv2
import argparse
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.dirname(os.path.abspath('..')))
from project_dir import project_dir




class Color_Name(nn.Module):
    def __init__(self):
        super(Color_Name, self).__init__()
        # color_steps = [0, 63, 127, 191, 255]
        # color_steps = [0, 85, 170, 255]
        color_steps = [0, 127, 255]
        colors =[]
        for i in range(len(color_steps)):
            for j in range(len(color_steps)):
                for k in range(len(color_steps)):
                    colors.append((color_steps[i], color_steps[j], color_steps[k]))

        self.color_dict = {}
        for i in range(len(colors)):
            self.color_dict[i] = colors[i]

        self.no_pixel = np.array([0, 0, 0]).astype('uint8')
        self.color_img = np.zeros([len(self.color_dict), 3, 1, 1])
        for k, v in self.color_dict.items():
            r, g, b = v
            self.color_img[k] = torch.stack([torch.ones(1, 1) * r, torch.ones(1, 1) * g, torch.ones(1, 1) * b])
        self.color_img = torch.from_numpy(self.color_img).cuda()


    def forward(self, img, mask_img):
        c, h, w = img.shape
        final_feature = torch.zeros((len(self.color_dict)))
        tmp_img = torch.zeros([len(self.color_dict), c, h, w])
        for i in range(len(self.color_dict)):
            tmp_img[i] = torch.abs(img - self.color_img[i])
        distance = torch.sum(tmp_img, dim=(1))
        feature = distance.argmin(dim=0)
        feature = feature - (1 - mask_img[0]) * len(self.color_dict)
        for i in range(len(self.color_dict)):
            final_feature[i] = torch.sum(feature == i)
        final_feature = (final_feature / final_feature.sum()).numpy()
        final_feature = [(i, feature) for (i, feature) in enumerate(final_feature)]
        return sorted(final_feature, key=lambda x: -x[1])


def mian():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    target_classes = os.listdir(args.root)

    for i, target_class in enumerate(target_classes):
        save_class_root = os.path.join(args.save_root, target_class)
        os.makedirs(save_class_root, exist_ok=True)
        train_files = os.listdir(os.path.join(args.root, target_class))
        for file in train_files:
            save_path = os.path.join(save_class_root, file)
            if os.path.exists(save_path):
                continue

            img_path = os.path.join(args.root, target_class, file)
            mask_path = os.path.join(args.mask_root, target_class, file)
            color_img = Image.open(img_path)
            color_img = color_img.convert('RGB')
            color_img = transform(color_img).cuda()*255
            mask_img = Image.open(mask_path)
            mask_img = mask_img.convert('RGB')
            mask_img = transform(mask_img)
            mask_img[mask_img > 0] = 1
            features = model_color(color_img, mask_img)

            save_img = []
            for i, feature in features:
                if feature > 0:
                    r, g, b = model_color.color_dict[i]
                    pixels = int(np.round(224*feature))
                    tmp_img = np.stack([np.ones([pixels, 224]) * b, np.ones([pixels, 224]) * g, np.ones([pixels, 224]) * r], axis=2)
                    try:
                        save_img = np.concatenate([save_img, tmp_img], axis=0)
                    except:
                        save_img = tmp_img.copy()
            cv2.imwrite(save_path, save_img)
            print('Finished %s'%i)
        print('Finished %s'%target_class)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate input images for phase scrambling')

    parser.add_argument('--root', help='path to GradCam processed images',
                        default='data/iLab/preprocessed_images/train/Segement_GradCam/img')
    parser.add_argument('--mask_root', help='path to GradCam processed mask',
                        default='data/iLab/preprocessed_images/train/Segement_GradCam/mask')
    parser.add_argument('--save_root', default='data/iLab/feature_images/color_block/train')

    args = parser.parse_args(sys.argv[1:])
    argv = parser.parse_args(sys.argv[1:])
    for k, v in vars(argv).items():
        try:
            if '/' in v:
                if not os.path.exists(v):
                    exec('args.' + k + ' = os.path.join(project_dir, v)')
        except:
            pass
        print(k, eval('args.' + k))

    model_color = Color_Name()
    mian()