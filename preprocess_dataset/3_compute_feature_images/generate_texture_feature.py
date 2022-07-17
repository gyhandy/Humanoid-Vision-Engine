import sys
import argparse
import os
sys.path.append(os.getcwd())
import cv2
import random
import numpy as np
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.dirname(os.path.abspath('..')))
from project_dir import project_dir



def get_candidate_patch(mask_img, step_num):
    step_y, step_x = [lens // step_num for lens in mask_img.shape]

    candidates = []
    for y in range(0, mask_img.shape[0] - step_y + 1, step_y):
        for x in range(0, mask_img.shape[1] - step_x + 1, step_x):
            candidate = mask_img[y:y + step_y, x:x + step_x].copy()
            if np.count_nonzero(candidate) / candidate.size > 0.99:
                candidates.append(candidate)
                # plt.imshow(candidate, cmap='gray')
                # plt.show()
    return candidates, step_y, step_x


def main():
    patch_num = 4
    target_classes = os.listdir(args.select_root)

    for target_class in target_classes:
        save_root = os.path.join(args.save_root, target_class)
        if os.path.exists(save_root):
            continue

        os.makedirs(save_root, exist_ok=True)
        img_root = os.path.join(args.img_root, 'img', target_class)
        mask_root = os.path.join(args.img_root, 'mask', target_class)
        img_files = os.listdir(os.path.join(args.select_root, target_class))

        for i, img_dir in enumerate(img_files):
            save_dir = os.path.join(save_root, img_dir)
            if os.path.exists(save_dir):
                continue
            img = cv2.imread(os.path.join(img_root, img_dir), 0)
            mask = cv2.imread(os.path.join(mask_root, img_dir), 0)

            if img.shape != mask.shape:
                range_y, range_x = np.where(img != 0)
            else:
                range_y, range_x = np.where(mask != 0)
            min_y, max_y = range_y.min(), range_y.max()
            min_x, max_x = range_x.min(), range_x.max()
            mask_img = cv2.resize(img[min_y:max_y, min_x:max_x].copy(), (224,224))
            # plt.imshow(mask_img, cmap='gray')
            # plt.show()
            try:
                for step_num in range(3, 8):
                    candidates, step_y, step_x = get_candidate_patch(mask_img, step_num)
                    if len(candidates) >= patch_num:
                        break
                random.seed(random_seed)
                candidates = random.sample(candidates, patch_num)
            except:
                continue

            texture_img = np.zeros([lens * 2 for lens in candidates[0].shape])
            steps = int(np.sqrt(patch_num))
            for j in range(steps):
                for k in range(steps):
                    texture_img[j*step_y:(j+1)*step_y, k*step_x:(k+1)*step_x] = candidates[j*steps+k]
            # plt.imshow(texture_img, cmap='gray')
            # plt.show()
            cv2.imwrite(save_dir, texture_img)
            print('saved %d %s!'%(i, target_class))


def parse_arguments(args):
  """Parses the arguments passed to the run.py script."""

  return parser.parse_args(args)



if __name__ == '__main__':
    random_seed = 9

    parser = argparse.ArgumentParser()
    parser.add_argument('--select_root', help='path to original images', type=str,
                        default='data/iLab/original_images/train')
    parser.add_argument('--img_root', type=str, help='path to GradCam segement results',
                        default='data/iLab/preprocessed_images/train/Segement_GradCam')
    parser.add_argument('--save_root', type=str,
                        default='data/iLab/feature_images/texture/train')
    args = parse_arguments(sys.argv[1:])
    argv = parser.parse_args(sys.argv[1:])
    for k, v in vars(argv).items():
        try:
            if '/' in v:
                if not os.path.exists(v):
                    exec('args.' + k + ' = os.path.join(project_dir, v)')
        except:
            pass
        print(k, eval('args.' + k))


    main()