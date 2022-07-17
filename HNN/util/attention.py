import numpy as np
import torch
from torch import nn
from torch.nn import init



class attention(nn.Module):

    def __init__(self, channel=2, class_num=1000, multi_layer=True):
        super().__init__()
        self.avgpool = nn.AvgPool2d(7, stride=1)

        if multi_layer:
            self.fc = nn.Sequential(
                nn.Linear(512*channel, 512, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(512, class_num, bias=False),
                nn.Softmax()
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(512*channel, class_num, bias=False),
                nn.Softmax()
            )


    def forward(self, latent_shape, latent_texture, latent_color=None, require_grad = False):
        try:
            x = torch.cat((latent_shape, latent_texture, latent_color), dim=1)
        except:
            x = torch.cat((latent_shape, latent_texture), dim=1)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        prediction = self.fc(x)
        if require_grad:
            index = np.argmax(prediction.cpu().data.numpy())
            target = prediction[0][index]
            target.backward()
            return latent_shape.grad, latent_texture.grad, latent_color.grad
        return prediction