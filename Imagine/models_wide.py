'''
Define Models: Discriminator and Generator
'''

import torch
import torch.nn as nn


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

    
class Double(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(Double, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, depth = 1, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()

        layers = []

        layers.append(nn.Conv2d(in_size, out_size, kernel_size=3,
                                padding=1))
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))

        layers.append(nn.LeakyReLU(0.2))

        if dropout:
            layers.append(nn.Dropout(dropout))

        for i in range(depth - 1):
            layers.append(nn.Conv2d(out_size, out_size, kernel_size=3,
                                    padding=1))
            if normalize:
                layers.append(nn.InstanceNorm2d(out_size))

            layers.append(nn.LeakyReLU(0.2))

            if dropout:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.MaxPool2d(2))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# class UNetDown_old(nn.Module):
#     def __init__(self, in_size, out_size, mid_channels=None, normalize=True, dropout=0.0):
#         super(UNetDown, self).__init__()

#         layers = []

#         layers.append(nn.Conv2d(in_size, out_size, kernel_size=4,
#                                 stride=2, padding=1, bias=False))
#         if normalize:
#             layers.append(nn.InstanceNorm2d(out_size))
#         layers.append(nn.LeakyReLU(0.2))
#         if dropout:
#             layers.append(nn.Dropout(dropout))

#         self.model = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size,  depth = 3, dropout=0.0):
        super(UNetUp, self).__init__()

        layers = [nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
                  nn.InstanceNorm2d(out_size),
                  nn.ReLU(inplace=True)]
        if dropout:
            layers.append(nn.Dropout(dropout))

        # deeper
        for i in range(depth - 1):
            layers.append(nn.Conv2d(out_size, out_size, kernel_size=3,
                                    padding=1))

            layers.append(nn.InstanceNorm2d(out_size))

            layers.append(nn.LeakyReLU(0.2))

            if dropout:
                layers.append(nn.ReLU(inplace=True))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x


class ShapeEncoder(nn.Module):
    def __init__(self, in_channels, out_channels=3):
        super(ShapeEncoder, self).__init__()

        self.down1 = UNetDown(in_channels, 64, depth = 3, normalize=False)
        self.down2 = UNetDown(64, 128, depth = 3 )
        self.down3 = UNetDown(128, 256, depth = 3)
        self.down4 = UNetDown(256, 512, depth = 3)

        self.down5 = UNetDown(512, 512, depth = 3, dropout=0.5)
        self.down6 = UNetDown(512, 512, depth = 3, dropout=0.5)

    def forward(self, x):
        d1 = self.down1(x) # 128
        d2 = self.down2(d1) # 64
        d3 = self.down3(d2) # 32
        d4 = self.down4(d3) # 16
        d5 = self.down5(d4) # 8
        d6 = self.down6(d5) # 4

        return (d1, d2, d3, d4, d5, d6)


class TextureEncoder(nn.Module):
    def __init__(self, in_channels):
        super(TextureEncoder, self).__init__()

        self.down1 = UNetDown(in_channels, 64, depth = 3, normalize=False)
        self.down2 = UNetDown(64, 128, depth = 3 )
        self.down3 = UNetDown(128, 256, depth = 3)
        self.down4 = UNetDown(256, 512, depth = 3)
        self.down5 = UNetDown(512, 512, depth = 3, dropout=0.5)
        self.down6 = UNetDown(512, 512, depth = 3, dropout=0.5)

    def forward(self, x):
        d1 = self.down1(x) # 128
        d2 = self.down2(d1) # 64
        d3 = self.down3(d2) # 32
        d4 = self.down4(d3) # 16
        d5 = self.down5(d4) # 8
        d6 = self.down6(d5) # 4

        return (d1, d2, d3, d4, d5, d6)


class ColorEncoder(nn.Module):
    def __init__(self, in_channels):
        super(ColorEncoder, self).__init__()

        self.down1 = UNetDown(in_channels, 64, depth = 3, normalize=False)
        self.down2 = UNetDown(64, 128, depth = 3 )
        self.down3 = UNetDown(128, 256, depth = 3)
        self.down4 = UNetDown(256, 512, depth = 3)
        self.down5 = UNetDown(512, 512, depth = 3, dropout=0.5)
        self.down6 = UNetDown(512, 512, depth = 3, dropout=0.5)

    def forward(self, x):
        d1 = self.down1(x) # 128
        d2 = self.down2(d1) # 64
        d3 = self.down3(d2) # 32
        d4 = self.down4(d3) # 16
        d5 = self.down5(d4) # 8
        d6 = self.down6(d5) # 4

        return (d1, d2, d3, d4, d5, d6)


# class ColorEncoder_linear(nn.Module):
#     def __init__(self, in_features):
#         super(ColorEncoder, self).__init__()

#         self.down1 = nn.Linear(in_features, 1024)
#         self.down2 = nn.Linear(1024, 256)
#         self.down3 = nn.Linear(256, 64)
#         self.down4 = nn.Linear(64, 16)

#     def forward(self, x):
#         d1 = self.down1(x)
#         d2 = self.down2(d1)
#         d3 = self.down3(d2)
#         d4 = self.down4(d3)

#         return (d1, d2, d3, d4)

class Fusion(nn.Module):
    def __init__(self, t_channel):
        super(Fusion, self).__init__()

        self.double_conv_s = Double(t_channel, t_channel)
        self.double_conv_t = Double(t_channel, t_channel)
        self.double_conv_c = Double(t_channel, t_channel)
        self.double_conv1 = Double(t_channel*2, t_channel)
        self.double_conv2 = Double(t_channel, t_channel)

        self.adain = adaptive_instance_normalization

        # self.upsample_1 = nn.Upsample(
        #     scale_factor=4, mode='bilinear', align_corners=True)
        # self.upsample_2 = nn.Upsample(
        #     scale_factor=4, mode='bilinear', align_corners=True)

    def forward(self, s, t, c):
        s = self.double_conv_s(s)
        t = self.double_conv_t(t)
        c = self.double_conv_c(c)

        x = torch.cat([s, t], dim = 1)

        x = self.double_conv1(x)

        x = self.adain(x, c)

        x = self.double_conv2(x)
        
        # deeper??

        return x

class Fusion_conv(nn.Module):
    def __init__(self, t_channel):
        super(Fusion, self).__init__()

        self.double_conv_s = Double(t_channel, t_channel)
        self.double_conv_t = Double(t_channel, t_channel)
        self.double_conv_c = Double(t_channel, t_channel)
        self.double_conv = Double(t_channel * 3, t_channel)


    def forward(self, s, t, c):
        s = self.double_conv_s(s)
        t = self.double_conv_t(t)
        c = self.double_conv_c(c)

        x = torch.cat([s, t, c], dim = 1)

        # AdaIN???
        x = self.double_conv(x)
        # deeper??

        return x

# class Fusion_linear_color(nn.Module):
#     def __init__(self, t_channel):
#         super(Fusion, self).__init__()

#         self.double_conv = Double(t_channel * 5 + 1, t_channel * 4)

#         self.upsample_1 = nn.Upsample(
#             scale_factor=4, mode='bilinear', align_corners=True)
#         self.upsample_2 = nn.Upsample(
#             scale_factor=4, mode='bilinear', align_corners=True)

#     def forward(self, s, t, c):
#         size = t.size(2)

#         t = self.upsample_1(t)

#         c = torch.reshape(c, (-1, 1, size, size))
#         c = self.upsample_2(c)

#         x = torch.cat([s, t, c], dim = 1)

#         x = self.double_conv(x)

#         return x


class Decoder(nn.Module):
    def __init__(self, out_channels=3):
        super(Decoder, self).__init__()

        self.fus1 = Fusion(64)
        self.fus2 = Fusion(128)
        self.fus3 = Fusion(256)
        self.fus4 = Fusion(512)
        self.fus5 = Fusion(512)
        self.fus6 = Fusion(512)

        self.double_conv = Double(512, 512)

        self.up1 = UNetUp(512, 512, depth = 3, dropout=0.5)
        self.up2 = UNetUp(1024, 512, depth = 3, dropout=0.5)
        self.up3 = UNetUp(1024, 256, depth = 3)
        self.up4 = UNetUp(512, 128, depth = 3)
        self.up5 = UNetUp(256, 64, depth = 3)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh()
        )

    def forward(self, s_list, t_list, c_list):

        (s1, s2, s3, s4, s5, s6) = s_list
        (t1, t2, t3, t4, t5, t6) = t_list
        (c1, c2, c3, c4, c5, c6) = c_list

        d1 = self.fus1(s1, t1, c1)
        d2 = self.fus2(s2, t2, c2)
        d3 = self.fus3(s3, t3, c3)
        d4 = self.fus4(s4, t4, c4)
        d5 = self.fus5(s5, t5, c5)
        d6 = self.fus6(s6, t6, c6)        
        # d5 = self.double(d4)

        u1 = self.up1(d6, d5) # (512 --> 256) + 512 = 1024
        u2 = self.up2(u1, d4) # (1024 --> 512) + 512 = 1024
        u3 = self.up3(u2, d3) # (1024 --> 64) + 64 = 128
        u4 = self.up4(u3, d2) # (256 --> 64) + 64 = 128
        u5 = self.up5(u4, d1) # (256 --> 64) + 64 = 128

        return self.final(u5)


class Generator(nn.Module):
    def __init__(self, shape_in_channels=1,
                 texture_in_channels=1,
                 out_channels=3):

        super(Generator, self).__init__()

        self.shape_encoder = ShapeEncoder(in_channels=shape_in_channels)
        self.texture_encoder = TextureEncoder(in_channels=texture_in_channels)
        self.color_encoder = ColorEncoder(in_channels=3)

        self.decoder = Decoder(out_channels=out_channels)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh()
        )


    def forward(self, s, t, c):
        s_emd = self.shape_encoder(s)
        t_emd = self.texture_encoder(t)
        c_emd = self.color_encoder(c)

        output = self.decoder(s_emd, t_emd, c_emd)

        return output


class Discriminator(nn.Module):
    '''Discriminator with PatchGAN'''

    def __init__(self, conv_dim, layer_num):
        super(Discriminator, self).__init__()

        layers = []

        # input layer
        layers.append(
            nn.Conv2d(3+1+1+3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        current_dim = conv_dim

        # hidden layers
        for i in range(1, layer_num):
            layers.append(nn.Conv2d(current_dim, current_dim*2,
                                    kernel_size=4, stride=2, padding=1))
            layers.append(nn.InstanceNorm2d(current_dim*2))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            current_dim *= 2

        self.model = nn.Sequential(*layers)

        # output layer
        self.conv_src = nn.Conv2d(
            current_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)

        self.upsample_1 = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample_2 = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=True)


    def forward(self, x, s, t, c):
        # t = self.upsample_1(t)

        # c = self.upsample_2(c)

        x = self.model(torch.cat([x, s, t, c], dim=1))
        out_src = self.conv_src(x)
        return out_src


if __name__ == '__main__':
    b = 4
    
    shape = torch.zeros((b, 1, 256, 256))
    texure = torch.zeros((b, 1, 256, 256))
    color = torch.zeros((b, 3, 256, 256))

    G = Generator()
    D = Discriminator(64, 6)

    x = G(shape, texure, color)

    print(x.shape)

    y = D(x, shape, texure, color)

    print(y.shape)


