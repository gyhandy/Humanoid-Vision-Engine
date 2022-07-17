import os
import cv2
import glob
import argparse
import sys
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.dirname(os.path.abspath('..')))
from project_dir import project_dir



parser = argparse.ArgumentParser()
parser.add_argument('--source_img_root', help='path to DPT processed images', type=str,
                    default='data/iLab/preprocessed_images/train/ori_img_monodepth')
parser.add_argument('--source_mask_root', type=str, help='path to GradCam segement mask',
                    default='data/iLab/preprocessed_images/train/Segement_GradCam/mask')
parser.add_argument('--save_root', type=str,
                    default='data/iLab/feature_images/shape/train')
argv = parser.parse_args(sys.argv[1:])
args = parser.parse_args(sys.argv[1:])
for k, v in vars(argv).items():
    try:
        if '/' in v:
            if not os.path.exists(v):
                exec('args.' + k + ' = os.path.join(project_dir, v)')
    except:
        pass
    print(k, eval('args.' + k))


target_classes = os.listdir(args.source_img_root)

for target_class in target_classes:
    img_path = os.path.join(args.source_img_root, target_class)
    mask_path = os.path.join(args.source_mask_root, target_class)
    save_path = os.path.join(args.save_root, target_class)
    os.makedirs(save_path, exist_ok=True)

    file_paths = glob.glob(os.path.join(img_path, "*.png"))
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        save_file_path = os.path.join(save_path, file_name)
        if os.path.exists(save_file_path):
            continue
        img = cv2.imread(file_path, 0)
        try:
            mask_dir = glob.glob(os.path.join(mask_path, "%s*")%file_name.replace('png', ''))[0]
            mask = cv2.imread(mask_dir, 0)/255
            multiply_img = mask * img
            cv2.imwrite(save_file_path, multiply_img)
            print('save %s %s'%(target_class, file_name))
        except:
            pass

    print('\n %s %d images!'%(target_class, len(os.listdir(save_path))))