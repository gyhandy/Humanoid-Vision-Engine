'''
Define Hyper-parameters
Init Dataset and Model
Run
'''

import argparse
import os

from solver import Solver
from loaders import Loaders


def main(config):

    # Environments Parameters
    os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda
    if not os.path.exists(config.output_path):
        os.makedirs(os.path.join(config.output_path, 'images/'))
        os.makedirs(os.path.join(config.output_path, 'models/'))
        os.makedirs(os.path.join(config.output_path, 'images_test/'))
    os.makedirs(os.path.join(config.output_path, 'result_mismatch/'))
        
        
    # Initialize Dataset
    loaders = Loaders(config)

    # Initialize Pixel2Pixel and train
    solver = Solver(config, loaders)
    print(config)
    if config.mode == "train":
        solver.train()
    elif config.mode == "test":
        solver.test()
    elif config.mode == "predict":
        solver.predict()
    else:
        print(config.mode, "should be train or test")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Environment Configuration
    parser.add_argument('--cuda', type=str, default='-1', help='If -1, use cpu; if >0 use single GPU; if 2,3,4 for multi GPUS(2,3,4)')
    parser.add_argument('--mode', type=str, default='predict')
    parser.add_argument('--data', type=str, default='texture') # 'all', testure, shape, color
    # parser.add_argument('--output_path', type=str, default='out/deeper_deeper_res_new_texture_unpaired/subset')
    # parser.add_argument('--output_path', type=str, default='out/deeper_deeper_res_new_texture/shape')
    # parser.add_argument('--output_path', type=str, default='out/deeper_deeper_res_new_texture/color')
    parser.add_argument('--output_path', type=str, default='out/deeper_deeper_res_new_texture/texture_new') # used
    # parser.add_argument('--output_path', type=str, default='out/deeper_deeper_res_new_texture/texture') 
    # parser.add_argument('--output_path', type=str, default='out/deeper_deeper/all')

    # Dataset Configuration
    # parser.add_argument('--dataset_path', type=str, default='/lab/tmpig8d/u/xingrui/GAN_data')
    # parser.add_argument('--dataset_path', type=str, default='/lab/tmpig8d/u/xingrui/GAN_texture_data')
    # res depth 9, adain
    # parser.add_argument('--dataset_path', type=str, default='/lab/tmpig8d/u/yao_data/human_simulation_engine/')
    # parser.add_argument('--dataset_path', type=str, default='/lab/tmpig8d/u/yao_data/human_simulation_engine/V3_shape_dataset')
    # parser.add_argument('--dataset_path', type=str, default='/lab/tmpig8d/u/yao_data/human_simulation_engine/V3_color_dataset')
    parser.add_argument('--dataset_path', type=str, default='/lab/tmpig8d/u/yao_data/human_simulation_engine/V3_texture_dataset')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--G_iter', type=int, default=5)

    # Model Configuration
    parser.add_argument('--conv_dim', type=int, default=64)
    parser.add_argument('--layer_num', type=int, default=6)

    # Train Configuration
    parser.add_argument('--resume_epoch', type=int, default=-1, help='if -1, train from scratch; if >=0, resume and start to train')
    parser.add_argument('--strict_load', type=bool, default=True)

    # Test Configuration
    parser.add_argument('--test_epoch', type=int, default=259)
    parser.add_argument('--mismatch', type=bool, default=True)
    parser.add_argument('--test_image', type=str, default='', help='if is an image, only translate it; if a folder, translate all images in it')

    # main function
    config = parser.parse_args()
    main(config)
