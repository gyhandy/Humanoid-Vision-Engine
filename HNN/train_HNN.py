import os,sys
from util.tools import *
from util.attention import attention
from util.data_loader import get_Dataloader
import torch
from torch import nn
import torch.optim as optim
import argparse
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('.'))
from project_dir import project_dir



def test(class_num, model_shape, model_texture, model_color, model_attention, test_loader, epoch, is_train, save_model_dir):
    model_attention.eval()

    class_num=class_num+1
    correct = [0] * class_num
    total = [0] * class_num
    acc = [0] * class_num
    for idx, (texture_img, shape_img, color_img, label, img_name) in enumerate(test_loader):
        texture_img = texture_img.to(device)
        shape_img = shape_img.to(device)
        color_img = color_img.to(device)
        label = label.to(device)
        img_class = img_name[0].split('_')[0]
        latent_shape = get_latent_output(model_shape, shape_img, 'resnet18')
        latent_texture = get_latent_output(model_texture, texture_img, 'resnet18')
        latent_color = get_latent_output(model_color, color_img, 'resnet18')
        output = model_attention(latent_shape, latent_texture, latent_color)
        _, pre = torch.max(output.data, 1)
        total[0] += label.size(0)
        pre = pre.squeeze()
        correct[0] += (pre == label).sum().item()
        for i in range(class_num-1):
            tmp = (torch.ones(label.size())) * i
            tmp = tmp.cuda()
            tmp = tmp.long()
            total[i+1] += (tmp == label).sum().item()
            correct[i+1] += ((tmp == label)*(pre == label)).sum().item()
    for i in range(class_num):
        try:
            acc[i] = correct[i]/total[i]
        except:
            acc[i] = 0
    log = open(os.path.join(save_model_dir, 'log.txt'), 'a')
    log.write("epoch "+str(epoch)+" in "+ is_train+":\n")
    log.write(str(acc))
    log.write('\n')
    log.close()
    print("epoch "+str(epoch)+" in "+ is_train+":")
    print(acc)
    return

def train(model_shape, model_texture, model_color, model_attention, train_loader, criterion, optimizer):
    model_attention.train()

    for idx, (texture_img, shape_img, color_img, label, _) in enumerate(train_loader):
        texture_img = texture_img.to(device)
        shape_img = shape_img.to(device)
        color_img = color_img.to(device)
        label = label.to(device)
        latent_shape = get_latent_output(model_shape, shape_img, 'resnet18')
        latent_texture = get_latent_output(model_texture, texture_img, 'resnet18')
        latent_color = get_latent_output(model_color, color_img, 'resnet18')
        output = model_attention(latent_shape, latent_texture, latent_color)
        loss = criterion(output,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.data)
    return


def main():
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


    class_num = len(os.listdir(os.path.join(args.root_shape, 'train')))
    root_shape = args.root_shape
    root_texture = args.root_texture
    root_color = args.root_color
    batch_size = args.batch_size
    model_shape = load_resnet18(class_num, args.shape_model)
    model_texture = load_resnet18(class_num, args.texture_model)
    model_color = load_resnet18(class_num, args.color_model)
    model_color = model_color.to(device)
    model_shape.eval()
    model_texture.eval()
    model_color.eval()
    train_loader = get_Dataloader(os.path.join(args.root_shape,'train'), os.path.join(args.root_texture,'train'), os.path.join(args.root_color,'train'), batch_size)
    test_loader = get_Dataloader(os.path.join(root_shape,'test'), os.path.join(root_texture,'test'), os.path.join(args.root_color,'test'), batch_size)
    model = attention(channel=3, class_num=class_num, multi_layer=False)
    if args.resume:
        model.load_state_dict(torch.load(args.resume))
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(),lr=args.lr,betas=args.betas)
    criterion = nn.CrossEntropyLoss().cuda()
    os.makedirs(args.save_model_dir, exist_ok=True)
    for epoch in range(args.epoch):
        train(model_shape,model_texture,model_color,model,train_loader, criterion, optimizer)
        test(class_num, model_shape, model_texture,model_color, model, train_loader, epoch, 'train', args.save_model_dir)
        test(class_num, model_shape, model_texture,model_color, model, test_loader, epoch, 'valid', args.save_model_dir)
        torch.save(model.state_dict(), os.path.join(args.save_model_dir,'model_ck_%d.pth'%epoch))
        with open(os.path.join(args.save_model_dir, 'log.txt'), 'a') as f:
            f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HNN Training')


    parser.add_argument('--resume', default="")

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
    parser.add_argument('--save_model_dir',
                        default='data/iLab/model/Attention')

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--betas', default=(0.9, 0.999), type=tuple)

    parser.add_argument('--epoch', default=10, type=int)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    main()