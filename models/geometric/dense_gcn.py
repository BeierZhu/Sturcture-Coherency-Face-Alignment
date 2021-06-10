import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdjacencyLearner(nn.Module):
    def __init__(self, in_channels, feat_size, num_points):
        super(AdjacencyLearner, self).__init__()
        self.op1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))
        self.op2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels*2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(in_channels*2),
            nn.ReLU(inplace=True))
        self.op3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels*2, out_channels=in_channels*2, kernel_size=2, padding=0),
            nn.BatchNorm2d(in_channels*2),
            nn.ReLU(inplace=True))
        self.op4 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels*2, out_channels=num_points**2, kernel_size=1, padding=0),
            nn.BatchNorm2d(num_points**2),
            nn.ReLU(inplace=True))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=num_points)

    def forward(self, x):
        x = self.op1(x)
        x = self.op2(x)
        x = self.op3(x)
        x = self.op4(x)
        x = self.pixel_shuffle(x)
        x = x.squeeze(1)
        x = (x + x.transpose(1, 2).contiguous())/2
        x = F.softmax(x, dim=1)
        return x


class DenseGConv(nn.Module):
    """
    graph convolution layer
    """

    def __init__(self, in_channels, out_channels, bias=True):
        super(DenseGConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # self.M = nn.Parameter(torch.eye(adj.size(0), dtype=torch.float), requires_grad=False)
        self.W = nn.Parameter(torch.zeros(size=(2, in_channels, out_channels), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, input, adj):
        h0 = torch.matmul(input, self.W[0])
        h1 = torch.matmul(input, self.W[1])

        M = torch.eye(adj.size(1), dtype=torch.float).to(input.device)
        output = torch.matmul(adj * M, h0) + torch.matmul(adj * (1 - M), h1)

        if self.bias is not None:
            return output + self.bias.view(1, 1, -1)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_channels) + ' -> ' + str(self.out_channels) + ')'
 

class DenseGCBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseGCBlock, self).__init__()
        self.g_conv = DenseGConv(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, adj):
        x = self.g_conv(x, adj).transpose(1, 2)
        x = self.bn(x).transpose(1, 2)
        x = self.relu(x)

        return x


class DynamicResGCBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hid_channels):
        super(DynamicResGCBlock, self).__init__()

        self.g_block1 = DenseGCBlock(in_channels, hid_channels)
        self.g_block2 = DenseGCBlock(hid_channels, out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)

    def forward(self, x_dict):
        x = x_dict['x']
        adj = x_dict['adj']
        residual = self.shortcut(x) if hasattr(self, 'shortcut') else x
        out = self.g_block1(x, adj)
        out = self.g_block2(out, adj)

        return {'x': out + residual, 'adj': adj}


class DenseGCNet(nn.Module):
    def __init__(self, in_channels, out_channels, hid_channels, num_layers=4):
        super(DenseGCNet, self).__init__()

        self.g_input = DenseGCBlock(in_channels, hid_channels)
        
        g_layers = []
        for _ in range(num_layers):
            g_layers.append(DynamicResGCBlock(hid_channels, hid_channels, hid_channels))

        self.g_layers = nn.Sequential(*g_layers)
        self.g_out = DenseGConv(hid_channels, out_channels)

    def forward(self, x, adj):
        out = self.g_input(x, adj)
        out = self.g_layers({'x': out, 'adj': adj})
        out = out['x']
        out = self.g_out(out, adj)

        return out


class MultiDynamicResGCBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hid_channels):
        super(MultiDynamicResGCBlock, self).__init__()

        self.g_block1 = DenseGCBlock(in_channels, hid_channels)
        self.g_block2 = DenseGCBlock(hid_channels, out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)

    def forward(self, x, adj):
        residual = self.shortcut(x) if hasattr(self, 'shortcut') else x
        out = self.g_block1(x, adj)
        out = self.g_block2(out, adj)

        return out + residual


class MultiDenseGCNet(nn.Module):
    def __init__(self, in_channels, out_channels, hid_channels):
        super(MultiDenseGCNet, self).__init__()

        self.g_input = DenseGCBlock(in_channels, hid_channels)

        self.g_layer1 = MultiDynamicResGCBlock(hid_channels, hid_channels, hid_channels)
        self.g_layer2 = MultiDynamicResGCBlock(hid_channels, hid_channels, hid_channels)
        self.g_layer3 = MultiDynamicResGCBlock(hid_channels, hid_channels, hid_channels)
        self.g_layer4 = MultiDynamicResGCBlock(hid_channels, hid_channels, hid_channels)

        self.g_out = DenseGConv(hid_channels, out_channels)

    def forward(self, x, adj_in, adj1, adj2, adj3, adj4, adj_out):
        out = self.g_input(x, adj_in)
        out = self.g_layer1(out, adj1)
        out = self.g_layer2(out, adj2)
        out = self.g_layer3(out, adj3)
        out = self.g_layer4(out, adj4)
        out = self.g_out(out, adj_out)

        return out