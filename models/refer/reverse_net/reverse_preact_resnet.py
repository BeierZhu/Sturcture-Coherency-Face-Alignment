import torch
import torch.nn as nn
from models.backbones.preact_resnet import PreActBlock
from models.backbones.preact_resnet import PreActResNet
from models.refinement import Refinement

from utils.log_helper import cprint

class PreActResNetBottomUp(PreActResNet):
    def __init__(self, block, num_blocks, num_feats, is_color=True):
        super(PreActResNetBottomUp, self).__init__(block, num_blocks, num_feats, is_color=True, receptive_keep=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        return out4, out3, out2, out1

class PreActResNetTopDown(nn.Module):
    def __init__(self, block, num_blocks, num_feats, num_boundary=1):
        self.in_planes = num_feats[3]*block.expansion
        super(PreActResNetTopDown, self).__init__()
        upscale = 2

        self.conv_sub = nn.Conv2d(in_channels=self.in_planes, out_channels=self.in_planes*upscale, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=upscale)
        self.relu = nn.ReLU(inplace=True)

        self.layer3 = self._make_layer(block, num_feats[2], num_blocks[2], stride=1)
        self.refine3 = Refinement(ku=num_feats[2]*block.expansion, kh=num_feats[2]*block.expansion, kd=num_feats[1]*block.expansion, upscale=upscale)

        self.layer2 = self._make_layer(block, num_feats[1], num_blocks[1], stride=1)
        self.refine2 = Refinement(ku=num_feats[1]*block.expansion, kh=num_feats[1]*block.expansion, kd=num_feats[0]*block.expansion, upscale=upscale)

        self.layer1 = self._make_layer(block, num_feats[0], num_blocks[0], stride=1)
        self.refine1 = Refinement(ku=num_feats[0]*block.expansion, kh=num_feats[0]*block.expansion, kd=num_feats[0]*block.expansion//2, upscale=1)

        self.conv = nn.Conv2d(in_channels=num_feats[0]*block.expansion//2, out_channels=num_boundary, kernel_size=3, padding=1)
        self._init_weights()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            self.in_planes = planes * block.expansion
            layers.append(block(self.in_planes, planes, stride))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')

    def forward(self, out4, out3, out2, out1):
        x = self.conv_sub(out4)
        x = self.pixel_shuffle(x)
        x = self.relu(x)

        x = self.layer3(x)
        x = self.refine3(x, out3)

        x = self.layer2(x)
        x = self.refine2(x, out2)

        x = self.layer1(x)
        x = self.refine1(x, out1)
        x = self.conv(x)

        return x

class PreActResUNet(nn.Module):
    def __init__(self, block, num_blocks, num_feats, is_color=True, num_boundary=1):
        super(PreActResUNet, self).__init__()
        self.bottom_up = PreActResNetBottomUp(block, num_blocks, num_feats, is_color=is_color)
        num_top_down_blocks = [1]*len(num_blocks)
        self.top_down = PreActResNetTopDown(block, num_top_down_blocks, num_feats,num_boundary)
        self.num_out_feats = self.bottom_up.num_out_feats
        self.downsample_ratio = self.bottom_up.downsample_ratio

    def forward(self, x):
        out4, out3, out2, out1 = self.bottom_up(x)
        heatmap = self.top_down(out4, out3, out2, out1)

        return out4, heatmap


def PreActResUNet18(is_color, pretrained_path=None, receptive_keep=False):
    cprint.green("Creating PreActResUNet18 ......")
    return PreActResUNet(block=PreActBlock, num_blocks=[2,2,2,2], is_color=is_color, num_feats=[64, 128, 256, 512])
    