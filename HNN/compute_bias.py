import os,sys
import numpy as np
from util.attention import attention
from util.data_loader import get_Dataloader
import torch
import argparse
import scipy
import json
from collections import defaultdict
from util.tools import *
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))
from project_dir import project_dir


def test(model_shape, model_texture, model_color, model_attention, data_loader, local_class='',  local_result={}):
    ratio_shape_list = []
    ratio_texture_list = []
    ratio_color_list = []
    for idx, (texture_img, shape_img, color_img, label, img_name) in enumerate(data_loader):
        if local_class and data_loader.dataset.label_list[label] != local_class:
            continue
        texture_img = texture_img.to(device)
        shape_img = shape_img.to(device)
        color_img = color_img.to(device)
        label = label.to(device)
        latent_shape = get_latent_output(model_shape, shape_img, 'resnet18')
        latent_texture = get_latent_output(model_texture, texture_img, 'resnet18')
        latent_color = get_latent_output(model_color, color_img, 'resnet18')
        grad_shape, grad_texture, grad_color = model_attention(torch.tensor(latent_shape, requires_grad=True),
                                                                             torch.tensor(latent_texture, requires_grad=True),
                                                                             torch.tensor(latent_color, requires_grad=True),
                                                                             require_grad = True)

        activation_shape = (latent_shape * grad_shape).relu().sum(axis=[1,2,3])
        activation_texture = (latent_texture * grad_texture).relu().sum(axis=[1,2,3])
        activation_color = (latent_color * grad_color).relu().sum(axis=[1,2,3])

        activation = torch.stack([activation_shape, activation_texture, activation_color], axis=1)
        activation = np.array([x.cpu().numpy() for x in activation])
        ratio_shape, ratio_texture, ratio_color = activation[:, 0], activation[:, 1], activation[:, 2]
        # ratio_shape, ratio_texture, ratio_color = F.softmax(activation)
        # ratio = activation / np.sum(activation, axis=1).reshape([-1,1])
        # ratio_shape, ratio_texture, ratio_color = ratio[:, 0], ratio[:, 1], ratio[:, 2]

        ratio_shape_list.extend(ratio_shape)
        ratio_texture_list.extend(ratio_texture)
        ratio_color_list.extend(ratio_color)


    norm_shape_list = []
    norm_texture_list = []
    norm_color_list = []
    for i in range(0, len(ratio_shape_list), args.norm_size):
        activation = [sum(x[i:i+args.norm_size]) for x in [ratio_shape_list, ratio_texture_list, ratio_color_list]]
        ratio_shape, ratio_texture, ratio_color = scipy.special.softmax(activation)
        norm_shape_list.append(ratio_shape)
        norm_texture_list.append(ratio_texture)
        norm_color_list.append(ratio_color)
        # print(ratio_shape, ratio_texture, ratio_color)

    ratio_shape, ratio_texture, ratio_color = [np.array(x).mean() for x in [norm_shape_list, norm_texture_list, norm_color_list]]
    ratio = np.stack([ratio_shape, ratio_texture, ratio_color], axis=0)
    # ratio_shape, ratio_texture, ratio_color = scipy.special.softmax(ratio)
    ratio_shape, ratio_texture, ratio_color = ratio / np.sum(ratio)
    print('%s shape ratio: %.4f' % (local_class, ratio_shape))
    print('%s texture ratio: %.4f' % (local_class, ratio_texture))
    print('%s color ratio: %.4f' % (local_class, ratio_color))

    local_result[local_class]['shape'] = ratio_shape
    local_result[local_class]['texture'] = ratio_texture
    local_result[local_class]['color'] = ratio_color

    with open('bias_results.json', 'w') as f:
        json.dump(local_result, f)
    return local_result


def main():
    class_num = len(os.listdir(os.path.join(args.root_shape, 'train')))
    root_shape = args.root_shape
    root_texture = args.root_texture
    root_color = args.root_color
    batch_size = args.batch_size
    model_shape = load_resnet18(class_num, args.shape_model)
    model_texture = load_resnet18(class_num, args.texture_model)
    model_color = load_resnet18(class_num, args.color_model)
    model_shape.eval()
    model_texture.eval()
    model_color.eval()
    train_loader = get_Dataloader(os.path.join(root_shape,'train'), os.path.join(root_texture,'train'), os.path.join(root_color,'train'), batch_size, shuffle=True)
    test_loader = get_Dataloader(os.path.join(root_shape,'test'), os.path.join(root_texture,'test'),os.path.join(root_color,'test'), batch_size, shuffle=True)
    model_attention = attention(channel=3, class_num=class_num)
    if args.attention_model_dir:
        model_attention.load_state_dict(torch.load(args.attention_model_dir))
    model_attention = model_attention.to(device).eval()
    label_list = test_loader.dataset.label_list
    local_result = defaultdict(dict)

    # global bias
    local_result = test(model_shape, model_texture, model_color, model_attention, test_loader, local_class='', local_result=local_result)

    # local bias
    # for target_class in label_list:
    #     local_result = test(model_shape, model_texture, model_color, model_attention, test_loader, local_class=target_class, local_result=local_result)
    #     print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='compute bias')

    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--norm_size', default=64, type=int)

    parser.add_argument('--root_shape',
                        default='data/iLab/feature_images/shape')
    parser.add_argument('--root_texture',
                        default='data/iLab/feature_images/texture')
    parser.add_argument('--root_color',
                        default='data/iLab/feature_images/color')
    parser.add_argument('--shape_model',
                        default='data/iLab/model/shape_resnet18/21.pth')
    parser.add_argument('--texture_model',
                        default='data/iLab/model/texture_resnet18/16.pth')
    parser.add_argument('--color_model',
                        default='data/iLab/model/color_resnet18/15.pth')
    parser.add_argument('--attention_model_dir',
                        default='data/iLab/model/Attention/model_ck0.pth')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    argv = parser.parse_args()
    for k, v in vars(argv).items():
        try:
            if '/' in v:
                if not os.path.exists(v):
                    exec('args.' + k + ' = os.path.join(project_dir, v)')
        except:
            pass
        print(k, eval('args.' + k))

    main()