import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.geometric.relation_learner import RelationLearner

class DynamicGConv(nn.Module):
    """
    graph convolution layer
    """

    def __init__(self, adj, in_channels, out_channels, bias=True):
        super(DynamicGConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.M = nn.Parameter(torch.eye(adj.size(0), dtype=torch.float), requires_grad=False)

        self.adj = adj
        self.m = (self.adj > 0)

        self.W = nn.Parameter(torch.zeros(size=(2, in_channels, out_channels), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, input, adj_weight):
        h0 = torch.matmul(input, self.W[0])
        h1 = torch.matmul(input, self.W[1])

        adj = -9e15 * torch.ones(input.size(0), self.adj.size(0), self.adj.size(1)).to(input.device)
        adj[:, self.m] = adj_weight
        adj = F.softmax(adj.view(-1, self.adj.size(0)), dim=1)
        adj = adj.view(-1, self.adj.size(0), self.adj.size(0))
        adj = adj.transpose(1, 2)

        import numpy as np
        print(adj)
        np.savetxt('a.txt', adj.squeeze(0).cpu().numpy(), fmt='%0.4f')
        import sys
        sys.exit(1)

        output = torch.matmul(adj * self.M, h0) + torch.matmul(adj * (1 - self.M), h1)

        if self.bias is not None:
            return output + self.bias.view(1, 1, -1)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_channels) + ' -> ' + str(self.out_channels) + ')'
 

class DynamicGCBlock(nn.Module):
    def __init__(self, adj, in_channels, out_channels):
        super(DynamicGCBlock, self).__init__()
        self.g_conv = DynamicGConv(adj, in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, adj):
        x = self.g_conv(x, adj).transpose(1, 2)
        x = self.bn(x).transpose(1, 2)
        x = self.relu(x)

        return x


class DynamicResGCBlock(nn.Module):
    def __init__(self, adj, in_channels, out_channels, hid_channels):
        super(DynamicResGCBlock, self).__init__()

        self.g_block1 = DynamicGCBlock(adj, in_channels, hid_channels)
        self.g_block2 = DynamicGCBlock(adj, hid_channels, out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)

    def forward(self, x_dict):
        x = x_dict['x']
        adj = x_dict['adj']
        residual = self.shortcut(x) if hasattr(self, 'shortcut') else x
        out = self.g_block1(x, adj)
        out = self.g_block2(out, adj)

        return {'x': out + residual, 'adj': adj}


class DynamicGCNetV2(nn.Module):
    def __init__(self, adj, in_channels, out_channels, hid_channels, num_points, top_k=3, num_layers=4):
        super(DynamicGCNetV2, self).__init__()

        self.relation_learner = RelationLearner(num_points*4, num_points, top_k=top_k) # TODO hard code here
        self.g_input = DynamicGCBlock(adj, in_channels, hid_channels) 
        self.num_layers = num_layers
        if num_layers != 0:
            g_layers = []
            for _ in range(num_layers):
                g_layers.append(DynamicResGCBlock(adj, hid_channels, hid_channels, hid_channels))

            self.g_layers = nn.Sequential(*g_layers)
        else:
            print('=========> zeros layers')

        self.g_out = DynamicGConv(adj, hid_channels, out_channels)

    def forward(self, x_in, x):
        adj = self.relation_learner(x_in)
        out = self.g_input(x, adj)
        if self.num_layers != 0:
            out = self.g_layers({'x': out, 'adj': adj})
            out = out['x']
        out = self.g_out(out, adj)

        return out

