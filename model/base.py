import torch
import torch.nn as nn

from efficientnet_pytorch import EfficientNet
from torch.nn.functional import dropout
from torchvision.models.resnet import resnet18

from .bev_deformable_attention import BEVDeformableTransformerEncoder
# import sys
# import os
# sys.path.append(os.path.abspath(__file__))
# from bev_deformable_attention import BEVDeformableTransformerEncoder

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1) # C上叠加
        return self.conv(x1)


class CamEncode(nn.Module):
    '''
    Bachbone
    '''
    def __init__(self, C):
        super(CamEncode, self).__init__()
        self.C = C

        self.trunk = EfficientNet.from_pretrained("efficientnet-b0") # Backbone主干
        self.up1 = Up(320+112, self.C) # 上采样

    def get_eff_depth(self, x):
        # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints)+1)] = prev_x
            prev_x = x

        # Head
        endpoints['reduction_{}'.format(len(endpoints)+1)] = x
        x = self.up1(endpoints['reduction_5'], endpoints['reduction_4'])
        return x

    def forward(self, x):
        return self.get_eff_depth(x)


class BevEncode(nn.Module):
    def __init__(self, inC, outC, instance_seg=True, embedded_dim=16, direction_pred=True, direction_dim=37):
        super(BevEncode, self).__init__()
        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64 + 256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

        self.instance_seg = instance_seg
        if instance_seg:
            self.up1_embedded = Up(64 + 256, 256, scale_factor=4)
            self.up2_embedded = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear',
                            align_corners=True),
                nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, embedded_dim, kernel_size=1, padding=0),
            )

        self.direction_pred = direction_pred
        if direction_pred:
            self.up1_direction = Up(64 + 256, 256, scale_factor=4)
            self.up2_direction = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear',
                            align_corners=True),
                nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, direction_dim, kernel_size=1, padding=0),
            )

    def forward(self, x):
        print(f"1: {x.shape}")
        # x: ([B, 64, 200, 400])
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x: ([B, 64, 100, 200])
        x1 = self.layer1(x)
        # print(f"2: {x1.shape}")
        # x1: ([B, 64, 100, 200])
        x = self.layer2(x1)
        x2 = self.layer3(x)
        # x2:([B, 256, 25, 50])
        # print(f"3: {x2.shape}")
        x = self.up1(x2, x1)
        
        x = self.up2(x)

        if self.instance_seg:
            x_embedded = self.up1_embedded(x2, x1)
            x_embedded = self.up2_embedded(x_embedded)
        else:
            x_embedded = None

        if self.direction_pred:
            x_direction = self.up1_embedded(x2, x1)
            x_direction = self.up2_direction(x_direction)
        else:
            x_direction = None
        
        # print(f"3: {x.shape}")
        # input()
        return x, x_embedded, x_direction

class BevTransformerEncoder(nn.Module):
    def __init__(self, inC, outC, num_layers=6, dropout=0.1, d_tf=256,
                 n_heads=8,instance_seg=True, embedded_dim=16, 
                 direction_pred=True, direction_dim=37):
        super(BevTransformerEncoder, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, d_tf, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(d_tf),
            nn.ReLU()
        )

        self.bev_transformer = BEVDeformableTransformerEncoder(num_layers,d_model=d_tf,dropout=dropout,activation="relu", n_heads=n_heads)
        
        self.up1 = Up(64 + d_tf, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

        self.instance_seg = instance_seg
        if instance_seg:
            self.up1_embedded = Up(64 + 256, 256, scale_factor=4)
            self.up2_embedded = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear',
                            align_corners=True),
                nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, embedded_dim, kernel_size=1, padding=0),
            )

        self.direction_pred = direction_pred
        if direction_pred:
            self.up1_direction = Up(64 + 256, 256, scale_factor=4)
            self.up2_direction = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear',
                            align_corners=True),
                nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, direction_dim, kernel_size=1, padding=0),
            )
    
    def forward(self, x):
        # print(f"1: {x.shape}")
        x1 = self.conv1(x)
        x = self.conv2(x1)
        # print(f"2: {x1.shape}")
        
        x2 = self.bev_transformer(x)
        # print(f"3: {x2.shape}")
        
        x = self.up1(x2, x1)
        
        x = self.up2(x)

        if self.instance_seg:
            x_embedded = self.up1_embedded(x2, x1)
            x_embedded = self.up2_embedded(x_embedded)
        else:
            x_embedded = None

        if self.direction_pred:
            x_direction = self.up1_embedded(x2, x1)
            x_direction = self.up2_direction(x_direction)
        else:
            x_direction = None
        
        # print(f"3: {x.shape}")
        # input()
        return x, x_embedded, x_direction

    
    
if __name__ == '__main__':
    in_c = 64
    model = BevTransformerEncoder(in_c, 4, num_layers=6, d_tf=256,n_heads=1)
    model.cuda()
    
    x = torch.randn(1, in_c, 200, 400).cuda()
    print(model(x)[0].shape) 
    
    
    # model = BevEncode(in_c, 4)
    # model.cuda()
    # x = torch.randn(1, in_c, 200, 400).cuda()
    # print(model(x)[0].shape) 
    