import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GConv(nn.Module):
    """
    graph convolution layer
    """

    def __init__(self, adj, in_channels, out_channels, bias=True):
        super(GConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.adj = nn.Parameter(adj, requires_grad=False)
        self.M = nn.Parameter(torch.eye(adj.size(0), dtype=torch.float), requires_grad=False)
        self.W = nn.Parameter(torch.zeros(size=(2, in_channels, out_channels), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        h0 = torch.matmul(input, self.W[0])
        h1 = torch.matmul(input, self.W[1])

        output = torch.matmul(self.adj * self.M, h0) + torch.matmul(self.adj * (1 - self.M), h1)

        if self.bias is not None:
            return output + self.bias.view(1, 1, -1)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_channels) + ' -> ' + str(self.out_channels) + ')'
 

class GCBlock(nn.Module):
    def __init__(self, adj, in_channels, out_channels):
        super(GCBlock, self).__init__()
        self.g_conv = GConv(adj, in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.g_conv(x).transpose(1, 2)
        x = self.bn(x).transpose(1, 2)
        x = self.relu(x)

        return x


class ResGCBlock(nn.Module):
    def __init__(self, adj, in_channels, out_channels, hid_channels):
        super(ResGCBlock, self).__init__()

        self.g_block1 = GCBlock(adj, in_channels, hid_channels)
        self.g_block2 = GCBlock(adj, hid_channels, out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        residual = self.shortcut(x) if hasattr(self, 'shortcut') else x
        out = self.g_block1(x)
        out = self.g_block2(out)

        return out + residual


class GCNet(nn.Module):
    def __init__(self, adj, in_channels, out_channels, hid_channels, num_layers=4):
        super(GCNet, self).__init__()

        self.g_input = GCBlock(adj, in_channels, hid_channels)
        
        g_layers = []
        for _ in range(num_layers):
            g_layers.append(ResGCBlock(adj, hid_channels, hid_channels, hid_channels))

        self.g_layers = nn.Sequential(*g_layers)
        self.g_out = GConv(adj, hid_channels, out_channels)

    def forward(self, x):
        out = self.g_input(x)
        out = self.g_layers(out)
        out = self.g_out(out)

        return out

