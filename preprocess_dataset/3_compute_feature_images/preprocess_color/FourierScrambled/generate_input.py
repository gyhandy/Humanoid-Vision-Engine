from __future__ import print_function
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os, sys
import time
from time import time
import numpy as np
import cv2
import argparse
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('..'))))
from project_dir import project_dir



def mian(args):
    target_classes = os.listdir(args.data_root)

    for i, target_class in enumerate(target_classes):
        save_class_root = os.path.join(args.save_root, target_class)
        os.makedirs(save_class_root, exist_ok=True)
        train_files = os.listdir(os.path.join(args.data_root, target_class))
        for file in train_files:
            save_path = os.path.join(save_class_root, file)
            if os.path.exists(save_path):
                continue

            start = time()
            img_path = os.path.join(args.data_root, target_class, file)
            mask_path = os.path.join(args.mask_root, target_class, file)

            color_img = np.array(cv2.imread(img_path),dtype='uint8')
            mask_img = np.array(cv2.imread(mask_path),dtype='uint8')
            mask_img[mask_img > 0] = 1

            if color_img.shape != mask_img.shape:
                range_y, range_x = np.where(color_img[:,:,0] != 0)
                avg_value = (color_img.sum(axis=0).sum(axis=0)) / ((color_img != 0).sum()//3)
            else:
                range_y, range_x = np.where(mask_img[:,:,0] != 0)
                avg_value = (color_img.sum(axis=0).sum(axis=0)) / np.count_nonzero(mask_img[:,:,0])

            avg_value = np.array([int(x) for x in avg_value],dtype='uint8')
            min_y, max_y = range_y.min(), range_y.max()
            min_x, max_x = range_x.min(), range_x.max()

            save_img = color_img + (1-mask_img)*avg_value
            save_img = cv2.resize(save_img[min_y:max_y, min_x:max_x, :], (224, 224))

            cv2.imwrite(save_path, save_img)
        print('Finished %s!'%target_class)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate input images for phase scrambling')

    parser.add_argument('--data_root', help='path to GradCam processed images',
                        default='data/iLab/preprocessed_images/train/Segement_GradCam/img')
    parser.add_argument('--mask_root', help='path to GradCam processed mask',
                        default='data/iLab/preprocessed_images/train/Segement_GradCam/mask')
    parser.add_argument('--save_root', help='path to save input images',
                        default='data/iLab/preprocessed_images/train/ori_resize')
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

    mian(args)