import os

from Ava_attention import Ava_attention
from data_loader_modality import get_modality_Dataloader, get_retrival_Dataloader
import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
import torchvision.models as models
from tools import *
import argparse
from keras import backend as K
import json


class Modality_model(nn.Module):

    def __init__(self, class_num=1000):
        super().__init__()
        self.modality_net = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=2, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 32, kernel_size=2, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.AvgPool2d(7, stride=1),
        ).cuda()

        self.fc = nn.Sequential(
            nn.Linear(32, 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(16, class_num, bias=False),
            nn.Softmax()
        )


    def preprocess_embedding(self, latent, type):
        embedding = self.modality_net(latent)
        x = embedding.view(embedding.size(0), -1)
        prediction = self.fc(x)
        return embedding, prediction


    def forward(self, latent_1, latent_2, latent_3, type_1, type_2, type_3):
        embedding_1, prediction_1 = self.preprocess_embedding(latent_1, type_1)
        embedding_2, prediction_2 = self.preprocess_embedding(latent_2, type_2)
        embedding_3, prediction_3 = self.preprocess_embedding(latent_3, type_3)
        return embedding_1, embedding_2, embedding_3, prediction_1, prediction_2, prediction_3


def get_concept_latent(img_list,model_shape,model_texture,model_color):
    texture_img, shape_img, color_img, label = img_list
    latent_shape = get_latent_output(model_shape, shape_img, 'resnet18')
    latent_texture = get_latent_output(model_texture, texture_img, 'resnet18')
    latent_color = get_latent_output(model_color, color_img, 'resnet18')
    return latent_shape, latent_texture, latent_color, label


def record_acc(pre, label, total, correct, class_num):
    _, pre = torch.max(pre.data, 1)
    total[0] += label.size(0)
    pre = pre.squeeze()
    correct[0] += (pre == label).sum().item()
    for i in range(class_num - 1):
        tmp = (torch.ones(label.size())) * i
        tmp = tmp.cuda()
        tmp = tmp.long()
        total[i + 1] += (tmp == label).sum().item()
        correct[i + 1] += ((tmp == label) * (pre == label)).sum().item()
    return total, correct


def test_classify(class_num, model_shape,model_texture,model_color,model_modality, test_loader, epoch, is_train):
    model_modality.eval()
    class_num=class_num+1
    correct = [0] * class_num
    total = [0] * class_num
    acc = [0] * class_num
    for idx, (img_list_a, img_list_p, img_list_n) in enumerate(test_loader):
        latent_shape_a, latent_texture_a, latent_color_a, label_a = get_concept_latent(img_list_a,model_shape,model_texture,model_color)
        latent_shape_p, latent_texture_p, latent_color_p, label_p = get_concept_latent(img_list_p,model_shape,model_texture,model_color)
        latent_shape_n, latent_texture_n, latent_color_n, label_n = get_concept_latent(img_list_n,model_shape,model_texture,model_color)

        con_1, con_2 = test_modalities
        con_1_a, con_1_p, con_1_n, pre_a, pre_p, pre_n = model_modality(eval('latent_%s_a' % con_1),
                                                                        eval('latent_%s_p' % con_1),
                                                                        eval('latent_%s_n' % con_1),
                                                                        con_1, con_1, con_1)
        total, correct = record_acc(pre_a, label_a, total, correct, class_num)

        con_2_a, con_2_p, con_2_n, pre_a, pre_p, pre_n = model_modality(eval('latent_%s_a' % con_2),
                                                                        eval('latent_%s_p' % con_2),
                                                                        eval('latent_%s_n' % con_2),
                                                                        con_2, con_2, con_2)
        total, correct = record_acc(pre_a, label_a, total, correct, class_num)

    for i in range(class_num):
        try:
            acc[i] = correct[i]/total[i]
        except:
            acc[i] = 0
    log = open(os.path.join(args.result_dir, 'log.txt'), 'a')
    log.write("epoch "+str(epoch)+" in "+ is_train+":\n")
    log.write(str(acc))
    log.write('\n')
    log.close()
    print("epoch "+str(epoch)+" in "+ is_train+":")
    print(acc)
    return


def train(model_shape,model_texture,model_color,model_modality,train_loader, classification_criterion, triplet_criterion, optimizer):
    model_modality.train()
    for idx, (img_list_a, img_list_p, img_list_n) in enumerate(train_loader):
        latent_shape_a, latent_texture_a, latent_color_a, label_a = get_concept_latent(img_list_a,model_shape,model_texture,model_color)
        latent_shape_p, latent_texture_p, latent_color_p, label_p = get_concept_latent(img_list_p,model_shape,model_texture,model_color)
        latent_shape_n, latent_texture_n, latent_color_n, label_n = get_concept_latent(img_list_n,model_shape,model_texture,model_color)

        con_1, con_2 = test_modalities
        con_1_a, con_1_p, con_1_n, pre_a, pre_p, pre_n = model_modality(eval('latent_%s_a' % con_1),
                                                                        eval('latent_%s_p' % con_1),
                                                                        eval('latent_%s_n' % con_1),
                                                                        con_1, con_1, con_1)
        loss_classification = classification_criterion(pre_a, label_a)
        loss_classification += classification_criterion(pre_p, label_p)
        loss_classification += classification_criterion(pre_n, label_n)

        loss_triplet = triplet_criterion(con_1_a, con_1_p, con_1_n)
        loss_triplet += triplet_criterion(con_1_p, con_1_a, con_1_n)

        con_2_a, con_2_p, con_2_n, pre_a, pre_p, pre_n = model_modality(eval('latent_%s_a' % con_2),
                                                                        eval('latent_%s_p' % con_2),
                                                                        eval('latent_%s_n' % con_2),
                                                                        con_2, con_2, con_2)
        loss_classification += classification_criterion(pre_a, label_a)
        loss_classification += classification_criterion(pre_p, label_p)
        loss_classification += classification_criterion(pre_n, label_n)

        loss_triplet += triplet_criterion(con_2_a, con_2_p, con_2_n)
        loss_triplet += triplet_criterion(con_2_p, con_2_a, con_2_n)

        loss_triplet += triplet_criterion(con_1_a, con_2_a, con_2_n)
        loss_triplet += triplet_criterion(con_1_a, con_2_p, con_2_n)
        loss_triplet += triplet_criterion(con_1_p, con_2_a, con_2_n)
        loss_triplet += triplet_criterion(con_1_p, con_2_p, con_2_n)
        loss_triplet += triplet_criterion(con_1_n, con_2_n, con_2_a)
        loss_triplet += triplet_criterion(con_1_n, con_2_n, con_2_p)

        loss_triplet += triplet_criterion(con_2_a, con_1_a, con_1_n)
        loss_triplet += triplet_criterion(con_2_a, con_1_p, con_1_n)
        loss_triplet += triplet_criterion(con_2_p, con_1_a, con_1_n)
        loss_triplet += triplet_criterion(con_2_p, con_1_p, con_1_n)
        loss_triplet += triplet_criterion(con_2_n, con_1_n, con_1_a)
        loss_triplet += triplet_criterion(con_2_n, con_1_n, con_1_p)

        loss_triplet += triplet_criterion(con_2_a, con_1_a, con_2_n)
        loss_triplet += triplet_criterion(con_2_p, con_1_a, con_2_n)
        loss_triplet += triplet_criterion(con_2_a, con_1_p, con_2_n)
        loss_triplet += triplet_criterion(con_2_p, con_1_p, con_2_n)
        loss_triplet += triplet_criterion(con_2_n, con_1_n, con_2_a)
        loss_triplet += triplet_criterion(con_2_n, con_1_n, con_2_p)

        loss_triplet += triplet_criterion(con_1_a, con_2_a, con_1_n)
        loss_triplet += triplet_criterion(con_1_p, con_2_a, con_1_n)
        loss_triplet += triplet_criterion(con_1_a, con_2_p, con_1_n)
        loss_triplet += triplet_criterion(con_1_p, con_2_p, con_1_n)
        loss_triplet += triplet_criterion(con_1_n, con_2_n, con_1_a)
        loss_triplet += triplet_criterion(con_1_n, con_2_n, con_1_p)

        optimizer.zero_grad()
        loss = loss_classification + loss_triplet
        loss.backward()
        optimizer.step()
        print(loss_classification.data, loss_triplet.data)
    return model_modality


def find_nearest_latent(anchor_latents, find_latents, class_num, label_dict, anchor, find, epoch=None):
    class_num=class_num+1
    correct = [0] * class_num
    total = [0] * class_num
    acc = [0] * class_num

    for idx, anchor_latent in enumerate(anchor_latents):
        # distance = abs(find_latents - anchor_latent)
        # distance = torch.sum(distance, dim=(1))
        distance = torch.sum(torch.square(find_latents - anchor_latent), axis=-1)
        if anchor == find:
            find_latent = int(torch.topk(-distance, 2)[1][1])
        else:
            find_latent = int(distance.argmin(dim=0))
        anchor_label = label_dict[idx]
        find_label = label_dict[find_latent]

        total[0] += 1
        total[anchor_label + 1] += 1
        if anchor_label == find_label:
            correct[0] += 1
            correct[anchor_label + 1] += 1

    for i in range(class_num):
        try:
            acc[i] = correct[i]/total[i]
        except:
            acc[i] = 0

    print("%s dataset, use %s to find %s:"%(test_bias,anchor,find))
    print(total)
    print(correct)
    print(acc)

    if not epoch is None:
        with open(os.path.join(args.result_dir, 'log.txt'), 'a') as f:
            f.write("%s dataset, use %s to find %s:\n"%(test_bias,anchor,find))
            f.write(str(acc))
            f.write('\n')
    return


def retrievel(class_num,dataloader, model_shape,model_texture,model_color,model_modality, epoch=None):
    model_modality.eval()
    shape_latents = torch.zeros([len(dataloader), 32])
    texture_latents = torch.zeros([len(dataloader), 32])
    color_latents = torch.zeros([len(dataloader), 32])

    for idx, img_list in enumerate(dataloader):
        latent_shape_a, latent_texture_a, latent_color_a, label_a = get_concept_latent(img_list,model_shape,model_texture,model_color)
        shape_a, texture_a, color_a, pre_shape_a, pre_texture_a, pre_color_a = model_modality(latent_shape_a, latent_texture_a,
                                                                                              latent_color_a, 'shape', 'texture', 'color')

        shape_latents[idx] = shape_a.view(-1)
        texture_latents[idx] = texture_a.view(-1)
        color_latents[idx] = color_a.view(-1)

    for anchor in test_modalities:
        for find in test_modalities:
            find_nearest_latent(eval(anchor+'_latents'), eval(find+'_latents'),
                                class_num, dataloader.dataset.image_label, anchor, find, epoch=epoch)
            print()



def retrievel_new_class(train_loader, test_loader, model_shape, model_texture, model_color, model_modality):
    model_modality.eval()
    train_class_list = sorted(os.listdir(os.path.join(args.root_shape, 'train')))
    train_class_dict = {k: v for k, v in enumerate(train_class_list)}

    test_class_list = sorted(os.listdir(os.path.join(args.root_shape, 'test')))
    test_class_dict = {k: v for k, v in enumerate(test_class_list)}

    result = json.loads(json.dumps({}))
    for find in test_modalities:
        find_latents = torch.zeros([len(train_loader), 32])
        for idx, (img_list,image_name) in enumerate(train_loader):
            texture_img, shape_img, color_img, label = img_list
            find_latent = get_latent_output(eval('model_%s' % find), eval('%s_img' % find), 'resnet18')
            find_latent, _ = model_modality.preprocess_embedding(find_latent, find)
            find_latents[idx] = find_latent.view(-1)

        find_latents = find_latents.cuda()
        for anchor in test_modalities:
            for idx, (img_list,image_name) in enumerate(test_loader):
                texture_img, shape_img, color_img, label = img_list
                anchor_latent = get_latent_output(eval('model_%s'%anchor), eval('%s_img'%anchor), 'resnet18')
                anchor_latent, _ = model_modality.preprocess_embedding(anchor_latent, anchor)

                anchor_latent = anchor_latent.view(-1)
                distance = torch.sum(torch.square(find_latents - anchor_latent), axis=-1)
                find_classes = torch.topk(-distance, len(find_latents))[1]
                find_classes = [train_class_dict[train_loader.dataset.image_label[int(i)]] for i in find_classes]

                distance = sorted(distance.detach().cpu().numpy())

                avg_distance = {}
                class_count = {}
                # find_class_list = []
                # distance_list = []
                # for i in range(len(find_classes)):
                #     if not find_classes[i] in find_class_list:
                #         find_class_list.append(find_classes[i])
                #         distance_list.append(distance[i].item())
                for i in range(len(find_classes)):
                    if not find_classes[i] in avg_distance.keys():
                        avg_distance[find_classes[i]] = distance[i]
                        class_count[find_classes[i]] = 1
                    else:
                        avg_distance[find_classes[i]] = avg_distance[find_classes[i]] + distance[i]
                        class_count[find_classes[i]] = class_count[find_classes[i]] + 1
                for key in avg_distance.keys():
                    avg_distance[key] = avg_distance[key]/class_count[key]
                avg_distance = sorted(avg_distance.items(), key=lambda d:d[1], reverse=False)
                
                result_one = {'class':[i[0] for i in avg_distance],'distance':[i[1] for i in avg_distance]}
                result[image_name[0]] = result_one
                #zhix
                # print(image_name[0], 'use %s to find %s :'%(anchor, find))
                # print(find_class_list)
                # print(distance_list)
    output = json.dumps(result,indent=4)
    with open('shape.json','w') as f:
        f.write(output)
    # print(result)


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

    model_modality = Modality_model(class_num)
    if args.resume:
        model_modality.load_state_dict(torch.load(args.resume))
    model_modality = model_modality.cuda()

    if args.task == 'train':
        optimizer = optim.Adam(model_modality.parameters(),lr=args.lr,betas=args.betas)
        classification_criterion = nn.CrossEntropyLoss().cuda()
        triplet_criterion = nn.TripletMarginLoss(margin=3.0, p=2).cuda()
        os.makedirs(args.result_dir, exist_ok=True)

        train_loader = get_modality_Dataloader(os.path.join(root_shape, 'train'), os.path.join(root_texture, 'train'),
                                               os.path.join(root_color, 'train'), batch_size)

        for epoch in range(args.epoch):
            model_modality = train(model_shape,model_texture,model_color,model_modality,train_loader, classification_criterion, triplet_criterion, optimizer)
            # test_classify(class_num, model_shape,model_texture,model_color,model_modality, train_loader, epoch, 'train')
            # test_classify(class_num, model_shape,model_texture,model_color,model_modality, test_loader, epoch, 'valid')

            retrievel_train_loader = get_retrival_Dataloader(os.path.join(root_shape, 'train'),
                                                  os.path.join(root_texture, 'train'),
                                                  os.path.join(root_color, 'train'), batch_size=1, shuffle=False)

            # retrievel_test_loader = get_retrival_Dataloader(os.path.join(root_shape, 'valid'),
            #                                       os.path.join(root_texture, 'valid'),
            #                                       os.path.join(root_color, 'valid'), batch_size=1, shuffle=False)

            # with open(os.path.join(args.result_dir, 'log.txt'), 'a') as f:
            #     f.write("\nepoch " + str(epoch) + " in train:\n")
            # retrievel(class_num, retrievel_train_loader, model_shape, model_texture, model_color, model_modality, epoch)

            with open(os.path.join(args.result_dir, 'log.txt'), 'a') as f:
                f.write("\nepoch " + str(epoch) + " in test:\n")
            retrievel(class_num, retrievel_train_loader, model_shape, model_texture, model_color, model_modality, epoch)

            torch.save(model_modality.state_dict(), os.path.join(args.result_dir,'modality_%s_ck%02d.pth'%('_'.join(test_modalities), epoch)))

    elif args.task == 'retrieval':
        train_loader = get_retrival_Dataloader(os.path.join(root_shape, 'train'), os.path.join(root_texture, 'train'),
                                                         os.path.join(root_color, 'train'), batch_size=1, shuffle=False)

        test_loader = get_retrival_Dataloader(os.path.join(root_shape, 'test'), os.path.join(root_texture, 'test'),
                                              os.path.join(root_color, 'test'), batch_size=1, shuffle=False)

        retrievel_new_class(train_loader, test_loader, model_shape, model_texture, model_color, model_modality)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

    # test_bias = 'shape/texture/color/all'
    test_bias = 'Zero_Shot'
    # flag = 'pretrained/no_pretrained'
    flag = 'pretrained'
    # test_modalities = 'shape/texture/color'
    #zhix
    test_modalities = ['shape']
    parser.add_argument('--resume',
                        # default="/lab/tmpig8d/u/yao_data/human_simulation_engine/Zero_Shot/Retrieval/shape_color_margin_3/modality_shape_color_ck19.pth")
                        default="/lab/tmpig8d/u/yao_data/human_simulation_engine/Zero_Shot/Retrieval/shape_texture_margin_3/modality_shape_texture_ck19.pth")
                        # default="/lab/tmpig8d/u/yao_data/human_simulation_engine/Zero_Shot/Retrieval/texture_color_margin_3/modality_texture_color_ck19.pth")
    parser.add_argument('--task', help='train/retrieval', default='retrieval')
    parser.add_argument('--result_dir',
                        default='/lab/tmpig8d/u/yao_data/human_simulation_engine/Zero_Shot/Retrieval/%s_margin_3' % ('_'.join(test_modalities)))


    if test_bias == 'Zero_Shot':
        parser.add_argument('--root_shape',
                            default='/lab/tmpig8d/u/yao_data/human_simulation_engine/Zero_Shot/shape')
        parser.add_argument('--root_texture',
                            default='/lab/tmpig8d/u/yao_data/human_simulation_engine/Zero_Shot/texture')
        parser.add_argument('--root_color',
                            default='/lab/tmpig8d/u/yao_data/human_simulation_engine/Zero_Shot/color')
        if flag == 'pretrained':
            parser.add_argument('--shape_model',
                                default='/lab/tmpig8d/u/yao_data/human_simulation_engine/model/Zero_Shot/shape_resnet18/12.pth')
            parser.add_argument('--texture_model',
                                default='/lab/tmpig8d/u/yao_data/human_simulation_engine/model/Zero_Shot/texture_resnet18/12.pth')
            parser.add_argument('--color_model',
                                default='/lab/tmpig8d/u/yao_data/human_simulation_engine/model/Zero_Shot/color_resnet18/12.pth')

    elif test_bias == 'all':
        parser.add_argument('--root_shape',
                            default='/lab/tmpig8d/u/yao_data/human_simulation_engine/V3_all_dataset/shape')
        parser.add_argument('--root_texture',
                            default='/lab/tmpig8d/u/yao_data/human_simulation_engine/V3_all_dataset/texture')
        parser.add_argument('--root_color',
                            default='/lab/tmpig8d/u/yao_data/human_simulation_engine/V3_all_dataset/color')
        if flag == 'pretrained':
            parser.add_argument('--shape_model',
                                default='/lab/tmpig8d/u/yao_data/human_simulation_engine/model/V3_all_biased_data/shape_resnet18/24.pth')
            parser.add_argument('--texture_model',
                                default='/lab/tmpig8d/u/yao_data/human_simulation_engine/model/V3_all_biased_data/texture_resnet18/57.pth')
            parser.add_argument('--color_model',
                                default='/lab/tmpig8d/u/yao_data/human_simulation_engine/model/V3_all_biased_data/color_resnet18/89.pth')

    elif test_bias == 'shape':
        parser.add_argument('--root_shape',
                            default='/lab/tmpig8d/u/yao_data/human_simulation_engine/V3_shape_dataset/shape')
        parser.add_argument('--root_texture',
                            default='/lab/tmpig8d/u/yao_data/human_simulation_engine/V3_shape_dataset/texture')
        parser.add_argument('--root_color',
                            default='/lab/tmpig8d/u/yao_data/human_simulation_engine/V3_shape_dataset/color')
        if flag == 'no_pretrained':
            parser.add_argument('--shape_model',
                                default='/lab/tmpig8d/u/yao_data/human_simulation_engine/model/V3_shape_biased_data_no_pretrained/shape_resnet18_pretrained/49.pth')
            parser.add_argument('--texture_model',
                                default='/lab/tmpig8d/u/yao_data/human_simulation_engine/model/V3_shape_biased_data_no_pretrained/texture_resnet18_pretrained/96.pth')
            parser.add_argument('--color_model',
                                default='/lab/tmpig8d/u/yao_data/human_simulation_engine/model/V3_shape_biased_data_no_pretrained/color_resnet18/96.pth')
        elif flag == 'pretrained':
            parser.add_argument('--shape_model',
                                default='/lab/tmpig8d/u/yao_data/human_simulation_engine/model/V3_shape_biased_data/shape_resnet18_pretrained/99.pth')
            parser.add_argument('--texture_model',
                                default='/lab/tmpig8d/u/yao_data/human_simulation_engine/model/V3_shape_biased_data/texture_resnet18_pretrained/95.pth')
            parser.add_argument('--color_model',
                                default='/lab/tmpig8d/u/yao_data/human_simulation_engine/model/V3_shape_biased_data/color_resnet18_pretrained/99.pth')


    elif test_bias == 'texture':
        parser.add_argument('--root_shape',
                            default='/lab/tmpig8d/u/yao_data/human_simulation_engine/V3_texture_dataset/shape')
        parser.add_argument('--root_texture',
                            default='/lab/tmpig8d/u/yao_data/human_simulation_engine/V3_texture_dataset/texture')
        parser.add_argument('--root_color',
                            default='/lab/tmpig8d/u/yao_data/human_simulation_engine/V3_texture_dataset/color')
        if flag == 'no_pretrained':
            parser.add_argument('--shape_model',
                                default='/lab/tmpig8d/u/yao_data/human_simulation_engine/model/V3_texture_biased_data_no_pretrained/shape_resnet18_pretrained/99.pth')
            parser.add_argument('--texture_model',
                                default='/lab/tmpig8d/u/yao_data/human_simulation_engine/model/V3_texture_biased_data_no_pretrained/texture_resnet18_pretrained/63.pth')
            parser.add_argument('--color_model',
                                default='/lab/tmpig8d/u/yao_data/human_simulation_engine/model/V3_texture_biased_data_no_pretrained/color_resnet18/96.pth')
        elif flag == 'pretrained':
            parser.add_argument('--shape_model',
                                default='/lab/tmpig8d/u/yao_data/human_simulation_engine/model/V3_texture_biased_data/shape_resnet18_pretrained/99.pth')
            parser.add_argument('--texture_model',
                                default='/lab/tmpig8d/u/yao_data/human_simulation_engine/model/V3_texture_biased_data/texture_resnet18_pretrained/99.pth')
            parser.add_argument('--color_model',
                                default='/lab/tmpig8d/u/yao_data/human_simulation_engine/model/V3_texture_biased_data/color_resnet18_pretrained/98.pth')

    elif test_bias == 'color':
        parser.add_argument('--root_shape',
                            default='/lab/tmpig8d/u/yao_data/human_simulation_engine/V3_color_dataset/shape')
        parser.add_argument('--root_texture',
                            default='/lab/tmpig8d/u/yao_data/human_simulation_engine/V3_color_dataset/texture')
        parser.add_argument('--root_color',
                            default='/lab/tmpig8d/u/yao_data/human_simulation_engine/V3_color_dataset/color')
        if flag == 'no_pretrained':
            parser.add_argument('--shape_model',
                                default='/lab/tmpig8d/u/yao_data/human_simulation_engine/model/V3_color_biased_data_no_pretrained/shape_resnet18_pretrained/70.pth')
            parser.add_argument('--texture_model',
                                default='/lab/tmpig8d/u/yao_data/human_simulation_engine/model/V3_color_biased_data_no_pretrained/texture_resnet18_pretrained/71.pth')
            parser.add_argument('--color_model',
                                default='/lab/tmpig8d/u/yao_data/human_simulation_engine/model/V3_color_biased_data_no_pretrained/color_resnet18/99.pth')
        elif flag == 'pretrained':
            parser.add_argument('--shape_model',
                                default='/lab/tmpig8d/u/yao_data/human_simulation_engine/model/V3_color_biased_data/shape_resnet18_pretrained/99.pth')
            parser.add_argument('--texture_model',
                                default='/lab/tmpig8d/u/yao_data/human_simulation_engine/model/V3_color_biased_data/texture_resnet18_pretrained/99.pth')
            parser.add_argument('--color_model',
                                default='/lab/tmpig8d/u/yao_data/human_simulation_engine/model/V3_color_biased_data/color_resnet18_pretrained/99.pth')

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--betas', default=(0.9, 0.999), type=tuple)
    parser.add_argument('--epoch', default=20, type=int)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()