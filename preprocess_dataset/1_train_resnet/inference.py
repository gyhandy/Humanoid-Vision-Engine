import argparse
import os, sys
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))
import torchvision.models as models

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from data_loader import data_loader
from helper import AverageMeter, save_checkpoint, accuracy, adjust_learning_rate
import resnet
from aux_bn import MixBatchNorm2d

import numpy as np
import cv2
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"



parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')


# split = 'ori/shape/texture/color/texture_max_box'
split = 'color'


# parser.add_argument('--data',default='/lab/tmpig8d/u/yao_data/human_simulation_engine/compute_bias_for_dataset/iLab/%s'%(split), help='path to dataset')
parser.add_argument('--data',default='/lab/tmpig8d/u/yao_data/human_simulation_engine/ilab2M_dataset', help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='/lab/tmpig8d/u/yao_data/human_simulation_engine/compute_bias_for_dataset/iLab/model/%s_resnet18/'%(split))
parser.add_argument('-p', '--pretrained', type=bool, default=False)
parser.add_argument('-b', '--batch-size', default=32, type=int, help='mini-batch size (default: 256)')
parser.add_argument('--resume', type=str, help='path to latest checkpoitn, (default: None)',
                    default='/lab/tmpig8d/u/yao_data/human_simulation_engine/model/ilab2M/resnet188.pth')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful to restarts)')

parser.add_argument('--epochs', default=100, type=int, help='numer of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, help='Weight decay (default: 1e-4)')
parser.add_argument('-j', '--workers', default=0, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('-m', '--pin-memory', dest='pin_memory', action='store_true', help='use pin memory')
parser.add_argument('--print-freq', '-f', default=10, type=int, help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', type=bool, default=True, help='evaluate model on validation set')

best_prec1 = 0.0

def load_resnet50():
    norm_layer = MixBatchNorm2d
    model = models.__dict__['resnet50'](num_classes=1000, norm_layer=norm_layer)

    checkpoint = torch.load(args.resume)
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


def load_resnet18():
    model = models.resnet18(pretrained=args.pretrained)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, class_num)
    # optionlly resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
    return model.cuda()


def main():
    global args, best_prec1, class_num
    args = parser.parse_args()
    # Data loading
    train_loader, val_loader = data_loader(args.data, args.batch_size, args.workers, args.pin_memory)
    class_num = len(train_loader.dataset.classes)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
    else:
        print("=> creating model '{}'".format(args.arch))
    os.makedirs(args.arch, exist_ok=True)

    if 'resnet18' in args.arch:
        model = load_resnet18()
    elif 'resnet50' in args.arch:
        model = load_resnet50()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # define loss and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    if os.path.isfile(args.resume):
        optimizer.load_state_dict(torch.load(args.resume)['optimizer'])

    # if args.evaluate:
    #     validate(train_loader2,val_loader, model, criterion,59)
    #     return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr)

        # train for one epoch
        if not args.evaluate:
            train(train_loader, model, criterion, optimizer, epoch, args.print_freq)
            validate(train_loader, model, epoch, 'train')
            validate(val_loader, model, epoch, 'valid')
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict()
            }, args.arch + str(epoch) + '.pth')
        else:
            validate(val_loader, model, epoch, 'valid')
            break

def train(train_loader, model, criterion, optimizer, epoch, print_freq):
    losses = AverageMeter()
    top1 = AverageMeter()
    # switch to train mode
    model.train()

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time

        target = target.cuda()
        input = input.cuda()
        output = model(input)
        loss = criterion(output, target)

        prec1, _ = accuracy(output.data, target, topk=(1,1))
        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        top1.update(prec1[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                epoch, i, len(train_loader), loss=losses, top1=top1))


def validate(val_loader, model, epoch, flag):
    model.eval()
    correct = [0] * (class_num + 1)
    total = [0] * (class_num + 1)
    acc = [0] * (class_num + 1)
    for (img,label) in val_loader:
        img = img.cuda()
        label = label.cuda()
        out = model(img)
        _, pre = torch.max(out.data, 1)
        total[0] += label.size(0)
        pre = pre.squeeze()
        correct[0] += (pre == label).sum().item()
        for i in range(class_num):
            tmp = (torch.ones(label.size())) * i
            tmp = tmp.cuda()
            tmp = tmp.long()
            total[i+1] += (tmp == label).sum().item()
            correct[i+1] += ((tmp == label)*(pre == label)).sum().item()

    for i in range(class_num + 1):
        try:
            acc[i] = correct[i]/total[i]
        except:
            acc[i] = 0
    print('{} accuracy: {}'.format(flag, correct[0] / total[0]))
    print(str(total))
    print(str(correct))
    log = open(os.path.join(args.arch, 'log.txt'), 'a')
    log.write("epoch "+str(epoch)+" in %s:\n"%flag)
    log.write(str(acc))
    log.write('\n')
    if flag == 'valid':
        log.write('\n')
    log.close()


if __name__ == '__main__':
    main()