import numpy as np
import torch
import torchvision.models as models
from util.aux_bn import MixBatchNorm2d, to_mix_status, to_clean_status, to_adv_status
from torch import nn
import os

def load_resnet18(class_num, path):
    model = models.resnet18()
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, class_num)

    # optionlly resume from a checkpoint
    if path:
        if os.path.isfile(path):
            print("=> loading checkpoint '{}'".format(path))
            checkpoint = torch.load(path)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(path, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(path))
    return model.cuda()

def load_resnet50(path):
    norm_layer = MixBatchNorm2d
    model = models.__dict__['resnet50'](num_classes=1000, norm_layer=norm_layer)

    checkpoint = torch.load(path)
    if 'state_dict' not in checkpoint:  # for loading cutmix resnext101 model
        raw_ckpt = checkpoint
        checkpoint = {'state_dict': raw_ckpt}

    already_mixbn = False
    for key in checkpoint['state_dict']:
        if 'aux_bn' in key:
            already_mixbn = True
            break

    if not already_mixbn:
        to_merge = {}
        for key in checkpoint['state_dict']:
            if 'bn' in key:
                tmp = key.split("bn")
                aux_key = tmp[0] + 'bn' + tmp[1][0] + '.aux_bn' + tmp[1][1:]
                to_merge[aux_key] = checkpoint['state_dict'][key]
            elif 'downsample.1' in key:
                tmp = key.split("downsample.1")
                aux_key = tmp[0] + 'downsample.1.aux_bn' + tmp[1]
                to_merge[aux_key] = checkpoint['state_dict'][key]
        checkpoint['state_dict'].update(to_merge)
    try:
        model.load_state_dict(checkpoint['state_dict'])
    except:
        new_state_dict = model.state_dict()
        for k,v in checkpoint['state_dict'].items():
        	new_state_dict[k[7:]] = v

        model.load_state_dict(new_state_dict)
    return model.cuda()

def get_latent_output(model, inputs, type):
    features = []

    def hook(module, input, output):
      features.append(output.clone().detach())

    if type == 'resnet50':
        try:
            handle = model.module.layer4[-1].conv3.register_forward_hook(hook)
        except:
            handle = model.layer4[-1].conv3.register_forward_hook(hook)
    elif type == 'resnet18':
        try:
            handle = model.module.layer4[-1].conv2.register_forward_hook(hook)
        except:
            handle = model.layer4[-1].conv2.register_forward_hook(hook)

    with torch.no_grad():
        y = model(inputs)
    handle.remove()
    output = features[0]
    # if len(output.shape) == 4:
    #   output = output.permute(0, 2, 3, 1)
    return output