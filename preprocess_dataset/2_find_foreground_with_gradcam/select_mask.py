import sys
import argparse
import os
import torch.nn as nn
import torch
import torchvision.models as models
import cv2
import numpy as np
from matplotlib import pyplot as plt
import json
import torch.nn.functional as F
import copy
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.dirname(os.path.abspath('..')))
from project_dir import project_dir



class GradCAM(object):
    def __init__(self, net, layer_name):
        self.net = net
        self.layer_name = layer_name
        self.feature = None
        self.gradient = None
        self.net.eval()
        self.handlers = []
        self._register_hook()

    def _get_features_hook(self, module, input, output):
        self.feature = output
        # print("feature shape:{}".format(output.size()))

    def _get_grads_hook(self, module, input_grad, output_grad):
        """
        :param input_grad: tuple, input_grad[0]: None
                                   input_grad[1]: weight
                                   input_grad[2]: bias
        :param output_grad:tuple
        :return:
        """
        self.gradient = output_grad[0]

    def _register_hook(self):
        for (name, module) in self.net.named_modules():
            if name == self.layer_name:
                self.handlers.append(module.register_forward_hook(self._get_features_hook))
                self.handlers.append(module.register_backward_hook(self._get_grads_hook))

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

    def __call__(self, inputs, index):
        """
        :param inputs: [1,3,H,W]
        :param index: class id
        :return:
        """
        self.net.zero_grad()
        output = self.net(inputs)  # [1,num_classes]
        if index is None:
            index = np.argmax(output.cpu().data.numpy())
        target = output[0][index]
        target.backward()

        gradient = self.gradient[0].cpu().data.numpy()  # [C,H,W]
        weight = np.mean(gradient, axis=(1, 2))  # [C]

        feature = self.feature[0].cpu().data.numpy()  # [C,H,W]

        cam = feature * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
        cam = np.sum(cam, axis=0)  # [H,W]
        cam = np.maximum(cam, 0)  # ReLU

        cam -= np.min(cam)
        cam /= np.max(cam)
        # resize to 224*224
        cam = cv2.resize(cam, inputs.shape[2:][::-1], cv2.INTER_LINEAR)
        return cam


class GradCAM_ResNet:
    # Adapted with some modification from https://www.pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/
    def __init__(self, model, gradcam_layer=None):
        """
        model: pre-softmax layer (logit layer)
        """
        self.model = model
        if not gradcam_layer is None:
            self.layerName = gradcam_layer
        else:
            self.layerName = self.find_target_layer()
        self.grad_cam = GradCAM(self.model, self.layerName)


    def find_target_layer(self):
        layer_name = None
        for name, m in self.model.named_modules():
            if isinstance(m, nn.Conv2d):
                layer_name = name
        if layer_name is None:
            raise ValueError("Could not find conv2d layer. Cannot apply GradCAM")
        return layer_name


    def compute_heatmap(self, x, classIdx, keep_percent=30):
        cam = self.grad_cam(x, classIdx)  # cam mask
        cam3 = np.expand_dims(cam, axis=2)
        cam3 = np.tile(cam3, [1, 1, 3])
        cam = np.where(cam3 > np.percentile(cam3, 100 - keep_percent), 1, 0)
        return cam3, cam


    def overlay_gradCAM(self, img, cam3, cam):
        new_img = cam * img
        new_img = new_img.astype("uint8")

        cam3 = np.uint8(255 * cam3)
        cam3 = cv2.applyColorMap(cam3, cv2.COLORMAP_JET)
        new_img_concat = 0.3 * cam3 + 0.5 * img
        new_img_concat = (new_img_concat * 255.0 / new_img_concat.max()).astype("uint8")

        return new_img, new_img_concat


    def showCAMs(self, img, x, chosen_class, keep_percent):
        plt.imshow(img.astype("uint8"))
        plt.axis('off')
        plt.show()
        cv2.imwrite('0.png', img.astype("uint8")[:, :, ::-1])

        cam3, cam = self.compute_heatmap(x=x.cuda(), classIdx=chosen_class, keep_percent=keep_percent)
        new_img, new_img_concat = self.overlay_gradCAM(img, cam3, cam)
        plt.imshow(new_img)
        plt.axis('off')
        plt.show()
        cv2.imwrite('1.png', new_img.astype("uint8")[:, :, ::-1])

        new_img_concat = cv2.cvtColor(new_img_concat, cv2.COLOR_BGR2RGB)
        plt.imshow(new_img_concat)
        plt.axis('off')
        plt.show()
        cv2.imwrite('2.png', new_img_concat.astype("uint8")[:, :, ::-1])


def get_preds(model, img):
    if len(img.shape) == 2:
        img = np.tile(img, [3, 1, 1])

    if img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    x = np.float32(img.copy().transpose(2, 0, 1)) / 255
    x = np.ascontiguousarray(x)
    x = x[np.newaxis, ...]
    x = torch.tensor(x, requires_grad=True)
    try:
        preds = model(x)
    except:
        preds = model(x.cuda())
    preds = F.softmax(preds)
    return img, x, preds



def select_by_GradCam(model, img_root, class_label, target_class, mask_root, gradCAM, img_files, target_id):
    for i, img_dir in enumerate(img_files):
        save_img_dir = os.path.join(args.save_root, 'Segement_GradCam', 'img', target_id, img_dir)
        save_mask_dir = os.path.join(args.save_root, 'Segement_GradCam', 'mask', target_id, img_dir)
        if os.path.exists(save_img_dir):
            continue

        try:
            with torch.no_grad():
                img, x, preds = get_preds(model, plt.imread(os.path.join(img_root, img_dir)))
            pre_class = preds[0].argmax()
            # if pre_class != class_label:
            #     print(target_class, i)
            #     continue

            masks = np.load(os.path.join(mask_root, img_dir + '.npy'), allow_pickle=True).astype("int")
            max_score = -1 * float('inf')
            gradCAM.showCAMs(img, x, pre_class, 30)
            cam3, cam = gradCAM.compute_heatmap(x=x.cuda(), classIdx=pre_class, keep_percent=10)

            for j in range(1, masks.max() + 1):
                mask = copy.deepcopy(masks)
                mask[mask != j] = 0
                mask[mask == j] = 1
                if np.count_nonzero(mask) <= mask.size / 100:
                    continue
                mask_img = np.stack([mask, mask, mask], axis=2) * img

                # with torch.no_grad():
                #     mask_img, x, preds = get_preds(model, mask_img)
                # if preds[0][class_label] <= np.percentile(preds[0].cpu().numpy(), 80):
                #     continue
                score = (mask * cam[:, :, 0]).sum(axis=(0, 1)) / np.count_nonzero(mask)

                if score > max_score:
                    max_score = score
                    output_mask_img = mask_img.copy()
                    output_mask = mask.copy() * 255

            output_mask = output_mask.reshape((mask.shape[0], -1, 1))
            output_mask = np.concatenate((output_mask, output_mask, output_mask), axis=2)

            plt.imshow(output_mask_img)
            plt.axis('off')
            plt.show()
            cv2.imwrite(save_img_dir, output_mask_img[:, :, ::-1])
            cv2.imwrite(save_mask_dir, output_mask)
            print('finished %s %d image!' % (target_class, i))
        except:
            print('Wrong!!!', os.path.join(img_root, img_dir))


def load_resnet18(ck_dir, num_classes):
    model = models.resnet18(pretrained=True)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)

    # optionlly resume from a checkpoint
    if ck_dir:
        if os.path.isfile(ck_dir):
            print("=> loading checkpoint '{}'".format(ck_dir))
            checkpoint = torch.load(ck_dir)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(ck_dir, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(ck_dir))
    return model.cuda()



def main():
    if args.ck_dir:
        model = load_resnet18(args.ck_dir, args.num_classes)
    else:
        model = models.resnet50(pretrained=True).cuda()
    model.eval()
    gradCAM = GradCAM_ResNet(model=model)

    with open("./imagenet_labels.json", 'r') as f:
        cats = json.load(f)

    cats_dict = {}
    for k, v in cats.items():
        cats_dict[v[0]] = (int(k), v[1])

    class_dict = {}
    tmp_cats = cats.items()
    for k, v in tmp_cats:
        class_dict[v[1]+'_%s'%k] = v[0]

    cats_list = sorted(os.listdir(args.img_root))
    class_dict = {k:v for k,v in enumerate(cats_list)}


    for target_class, target_id in sorted(class_dict.items()):
        save_img_root = os.path.join(args.save_root, 'Segement_GradCam', 'img', target_id)

        os.makedirs(save_img_root, exist_ok=True)
        save_mask_root = os.path.join(args.save_root, 'Segement_GradCam', 'mask', target_id)
        os.makedirs(save_mask_root, exist_ok=True)
        img_root = os.path.join(args.img_root, target_id)
        mask_root = os.path.join(args.mask_root, target_id)
        img_files = os.listdir(img_root)
        # class_label = cats_dict[target_id][0]
        class_label = target_class

        select_by_GradCam(model, img_root, class_label, target_class, mask_root, gradCAM, img_files, target_id)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_root', type=str, help='path to original images',
                        default='data/iLab/original_images/train')
    parser.add_argument('--mask_root', type=str, help='path to open set segement results',
                        default='data/iLab/Segement/train')
    parser.add_argument('--ck_dir', type=str, help='the checkpoint used to show gradcam',
                        default='data/iLab/model/ori_resnet18/resnet188.pth')
    parser.add_argument('--num_classes', type=int, help='the number of classify classes', default=10)
    parser.add_argument('--save_root', type=str, help='path to GradCam processed results',
                        default='data/iLab/preprocessed_images/train')

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

    main()