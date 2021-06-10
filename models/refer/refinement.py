import torch
import torch.nn as nn


class Refinement(nn.Module):
    def __init__(self, ku, kh, kd, upscale):
        super(Refinement, self).__init__()
        self.upscale = upscale
        self.relu = nn.ReLU(inplace=True)

        self.conv_h = nn.Conv2d(in_channels=kh, out_channels=kh//2, kernel_size=3, padding=1)
        self.bn_h = nn.BatchNorm2d(kh//2)

        self.conv_u = nn.Conv2d(in_channels=ku, out_channels=ku//2, kernel_size=3, padding=1)
        self.bn_u = nn.BatchNorm2d(ku//2)

        self.conv_d = nn.Conv2d(in_channels=kh//2+ku//2, out_channels=kd, kernel_size=3, padding=1)
        self.bn_d = nn.BatchNorm2d(kd)

        if upscale != 1:
            self.conv_sub = nn.Conv2d(in_channels=kd, out_channels=kd*upscale*upscale, kernel_size=3, padding=1)
            self.pixel_shuffle = nn.PixelShuffle(upscale_factor=upscale)

    def forward(self, f_u, f_h):
        f_h = self.conv_h(f_h)
        f_h = self.bn_h(f_h)
        f_h = self.relu(f_h)

        f_u = self.conv_u(f_u)
        f_u = self.bn_u(f_u)
        f_u = self.relu(f_u)
        # concatenat
        x = torch.cat([f_u, f_h], 1)

        x = self.conv_d(x)
        x = self.bn_d(x)
        x = self.relu(x)

        # subpixel
        if self.upscale != 1: 
            x = self.conv_sub(x)
            x = self.pixel_shuffle(x)
            x = self.relu(x)
            
        return x