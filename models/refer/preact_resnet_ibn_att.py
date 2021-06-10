import torch
import torch.nn as nn
from models.backbones.preact_resnet import PreActBlock
from models.backbones.preact_resnet import PreActBottleneck
from models.backbones.preact_resnet import PreActResNet
from models.backbones.preact_resnet_ibn import PreActBlockIBN_a, PreActBottleneckIBN_a
from models.backbones.preact_resnet_ibn import PreActResNetIBN
from utils.log_helper import cprint

__all__ = ["PreActResNet18IBN_att", "PreActResNet34IBN_att", "PreActResNet50IBN_att"]

class PreActResNetIBN_att(PreActResNet):
    def __init__(self, blocks, num_blocks, num_feats, is_color=True, receptive_keep=False):
        super(PreActResNet, self).__init__()
        self.in_planes = num_feats[0]

        num_input_channel = 3 if is_color else 1

        self.conv1 = nn.Conv2d(num_input_channel, num_feats[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_feats[0], momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(blocks[0], num_feats[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(blocks[1], num_feats[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(blocks[2], num_feats[2], num_blocks[2], stride=2)
        stride = 1 if receptive_keep else 2
        self.layer4 = self._make_layer(blocks[3], num_feats[3], num_blocks[3], stride=stride)

        self.att1 = Spatial_att((256//4)**2,2)
        self.att2 = Spatial_att((256//8)**2,2)
        self.att3 = Spatial_att((256//16)**2,2)
        self.att4 = Spatial_att((256//(16*stride))**2,2)

        self.num_out_feats = [num_feat*blocks[3].expansion for num_feat in num_feats]
        self.downsample_ratio = 16*stride

        self._init_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        out1 = self.layer1(x)
        out1 = self.att1(out1)
        out2 = self.layer2(out1)
        out2 = self.att2(out2)
        out3 = self.layer3(out2)
        out3 = self.att3(out3)
        out4 = self.layer4(out3)
        out4 = self.att4(out4)

        x_dict = {'out1': out1, 'out2': out2, 'out3': out3, 'out4':out4}
        return x_dict


def PreActResNet18IBN_att(is_color, pretrained_path=None, receptive_keep=False, **kwargs):
    cprint.green("Creating PreActResNet18IBN_att ......")
    blocks = [PreActBlockIBN_a, PreActBlockIBN_a, PreActBlockIBN_a, PreActBlock]
    return PreActResNetIBN_att(blocks=blocks, num_blocks=[2,2,2,2], num_feats=[64, 128, 256, 512], 
                        is_color=is_color, receptive_keep=receptive_keep)

def PreActResNet34IBN_att(is_color, pretrained_path=None, receptive_keep=False, **kwargs):
    cprint.green("Creating PreActResNet34IBN_att ......")
    blocks = [PreActBlockIBN_a, PreActBlockIBN_a, PreActBlockIBN_a, PreActBlock]
    model = PreActResNetIBN_att(blocks=blocks, num_blocks=[3,4,6,3], 
                        num_feats=[64, 128, 256, 512], is_color=is_color, receptive_keep=receptive_keep)
    return model

def PreActResNet50IBN_att(is_color, pretrained_path=None, receptive_keep=False, **kwargs):
    cprint.green("Creating PreActResNet50IBN_att ......")
    blocks = [PreActBottleneckIBN_a, PreActBottleneckIBN_a, PreActBottleneckIBN_a, PreActBottleneck]
    model = PreActResNetIBN_att(blocks=blocks, num_blocks=[3,4,6,3], 
                        num_feats=[64, 128, 256, 512], is_color=is_color, receptive_keep=receptive_keep)
    return model


class PreActResNetIBN_HeatmapAtt(PreActResNet):
    def __init__(self, blocks, num_blocks, num_feats, is_color=True, receptive_keep=False, expansion=1):
        super(PreActResNet, self).__init__()
        self.in_planes = num_feats[0]

        num_input_channel = 3 if is_color else 1

        self.conv1 = nn.Conv2d(num_input_channel, num_feats[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_feats[0], momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(blocks[0], num_feats[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(blocks[1], num_feats[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(blocks[2], num_feats[2], num_blocks[2], stride=2)
        stride = 1 if receptive_keep else 2
        self.layer4 = self._make_layer(blocks[3], num_feats[3], num_blocks[3], stride=stride)
        self.receptive_keep = receptive_keep
        self.att = nn.Sequential(nn.Conv2d(num_feats[0]*expansion,256,kernel_size=3,padding=1,stride=1),
                                 nn.ReLU(True),
                                 nn.Conv2d(256,128,kernel_size=3,padding=1,stride=1),
                                 nn.ReLU(True),
                                 nn.Conv2d(128,64,kernel_size=3,padding=1,stride=1),
                                 nn.ReLU(True),
                                 nn.Conv2d(64,32,kernel_size=3,padding=1,stride=1),
                                 nn.ReLU(True),
                                 nn.Conv2d(32,1,kernel_size=1,padding=0,stride=1),
                                 nn.Sigmoid())
        self.avpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.num_out_feats = [num_feat*blocks[3].expansion for num_feat in num_feats]
        self.downsample_ratio = 16*stride

        self._init_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        out1 = self.layer1(x)
        att = self.att(out1)
        out2 = self.layer2(out1)
        att_down = self.avpool(att)
        out2 = out2*att_down + out2
        out3 = self.layer3(out2)
        att_down = self.avpool(att_down)
        out3 = out3*att_down + out3
        out4 = self.layer4(out3)
        if self.receptive_keep is False:
            att_down = self.avpool(att_down)
        out4 = out4*att_down + out4

        x_dict = {'out1': out1, 'out2': out2, 'out3': out3, 'out4':out4,'att': att}
        return x_dict


def PreActResNet18IBN_HeatmapAtt(is_color, pretrained_path=None, receptive_keep=False, **kwargs):
    cprint.green("Creating PreActResNet18IBN_HeatmapAtt ......")
    blocks = [PreActBlockIBN_a, PreActBlockIBN_a, PreActBlockIBN_a, PreActBlock]
    return PreActResNetIBN_HeatmapAtt(blocks=blocks, num_blocks=[2,2,2,2], num_feats=[64, 128, 256, 512], 
                        is_color=is_color, receptive_keep=receptive_keep,expansion=1)

def PreActResNet34IBN_HeatmapAtt(is_color, pretrained_path=None, receptive_keep=False, **kwargs):
    cprint.green("Creating PreActResNet34IBN_HeatmapAtt ......")
    blocks = [PreActBlockIBN_a, PreActBlockIBN_a, PreActBlockIBN_a, PreActBlock]
    model = PreActResNetIBN_HeatmapAtt(blocks=blocks, num_blocks=[3,4,6,3], 
                        num_feats=[64, 128, 256, 512], is_color=is_color, receptive_keep=receptive_keep,expansion=1)
    return model

def PreActResNet50IBN_HeatmapAtt(is_color, pretrained_path=None, receptive_keep=False, **kwargs):
    cprint.green("Creating PreActResNet50IBN_HeatmapAtt ......")
    blocks = [PreActBottleneckIBN_a, PreActBottleneckIBN_a, PreActBottleneckIBN_a, PreActBottleneck]
    model = PreActResNetIBN_HeatmapAtt(blocks=blocks, num_blocks=[3,4,6,3], 
                        num_feats=[64, 128, 256, 512], is_color=is_color, receptive_keep=receptive_keep,expansion=4)
    return model



class PreActResNetIBN_CHatt(PreActResNet):
    def __init__(self, blocks, num_blocks, num_feats, is_color=True, receptive_keep=False):
        super(PreActResNet, self).__init__()
        self.in_planes = num_feats[0]

        num_input_channel = 3 if is_color else 1

        self.conv1 = nn.Conv2d(num_input_channel, num_feats[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_feats[0], momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(blocks[0], num_feats[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(blocks[1], num_feats[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(blocks[2], num_feats[2], num_blocks[2], stride=2)
        stride = 1 if receptive_keep else 2
        self.layer4 = self._make_layer(blocks[3], num_feats[3], num_blocks[3], stride=stride)

        self.att1 = SELayer(num_feats[0],2)
        self.att2 = SELayer(num_feats[1],2)
        self.att3 = SELayer(num_feats[2],4)
        self.att4 = SELayer(num_feats[3],4)

        self.num_out_feats = [num_feat*blocks[3].expansion for num_feat in num_feats]
        self.downsample_ratio = 16*stride

        self._init_weights()

    def forward(self, x):
        x = self.conv1(x) # 128x128
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # 64x64

        out1 = self.layer1(x)
        out1 = self.att1(out1)
        out2 = self.layer2(out1)
        out2 = self.att2(out2)
        out3 = self.layer3(out2)
        out3 = self.att3(out3)
        out4 = self.layer4(out3)
        out4 = self.att4(out4)

        x_dict = {'out1': out1, 'out2': out2, 'out3': out3, 'out4':out4}
        return x_dict

def PreActResNet18IBN_CHatt(is_color, pretrained_path=None, receptive_keep=False, **kwargs):
    cprint.green("Creating PreActResNet18IBN_CHatt ......")
    blocks = [PreActBlockIBN_a, PreActBlockIBN_a, PreActBlockIBN_a, PreActBlock]
    return PreActResNetIBN_CHatt(blocks=blocks, num_blocks=[2,2,2,2], num_feats=[64, 128, 256, 512], 
                        is_color=is_color, receptive_keep=receptive_keep)


def PreActResNet50IBN_CHatt(is_color, pretrained_path=None, receptive_keep=False, **kwargs):
    cprint.green("Creating PreActResNet50IBN_CHatt ......")
    blocks = [PreActBlockIBN_a, PreActBlockIBN_a, PreActBlockIBN_a, PreActBlock]
    return PreActResNetIBN_CHatt(blocks=blocks, num_blocks=[3,4,6,3], 
                        num_feats=[64, 128, 256, 512], is_color=is_color, receptive_keep=receptive_keep)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)#.view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Spatial_att(nn.Module):
    def __init__(self, feat_size,reduction):
        super(Spatial_att, self).__init__()
        self.spatial_att = nn.Sequential(
            nn.Linear(feat_size, feat_size//reduction),
            nn.ReLU(inplace=True),
            nn.Linear(feat_size//reduction, feat_size),
            nn.Sigmoid())

    def forward(self, x):
        b, c, w, h = x.size()
        y = torch.mean(x,1).view(b,-1)
        y = self.spatial_att(y).view(b,1,w,h) # Bx1xWxH
        return x * y.expand_as(x)