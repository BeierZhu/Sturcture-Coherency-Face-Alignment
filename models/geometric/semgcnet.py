import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSemGConv(nn.Module):
    def __init__(self, adj, in_channels, out_channels, bias=True, num_head=4):
        super(MultiHeadSemGConv, self).__init__()
        num_head = min(out_channels, num_head)
        head_channels = out_channels//num_head
        self.head_sem = []
        for _ in range(num_head):
            self.head_sem.append(SemGConv(adj, in_channels, head_channels).cuda())

    def forward(self, x):
        return torch.cat([sem(x) for sem in self.head_sem], dim=2)

class MultiHeadSemGCBlock(nn.Module):
    def __init__(self, adj, in_channels, out_channels, num_head=4):
        super(MultiHeadSemGCBlock, self).__init__()
        self.sem_conv = MultiHeadSemGConv(adj, in_channels, out_channels, num_head)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.sem_conv(x).transpose(1, 2)
        x = self.bn(x).transpose(1, 2)
        x = self.relu(x)

        return x

class MultiHeadResSemGCBlock(nn.Module):
    def __init__(self, adj, in_channels, out_channels, hid_channels, num_head=4):
        super(MultiHeadResSemGCBlock, self).__init__()

        self.sem_block1 = MultiHeadSemGCBlock(adj, in_channels, hid_channels, num_head)
        self.sem_block2 = MultiHeadSemGCBlock(adj, hid_channels, out_channels, num_head)

        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        residual = self.shortcut(x) if hasattr(self, 'shortcut') else x
        out = self.sem_block1(x)
        out = self.sem_block2(out)

        return out + residual

class MultiHeadSemGCNet(nn.Module):
    def __init__(self, adj, in_channels, out_channels, hid_channels, num_layers=4, num_head=4):
        super(MultiHeadSemGCNet, self).__init__()

        self.sem_input = MultiHeadSemGCBlock(adj, in_channels, hid_channels, num_head)
        
        sem_layers = []
        for _ in range(num_layers):
            sem_layers.append(MultiHeadResSemGCBlock(adj, hid_channels, hid_channels, hid_channels, num_head))

        self.sem_layers = nn.Sequential(*sem_layers)
        self.sem_out = MultiHeadSemGConv(adj, hid_channels, out_channels, num_head)

    def forward(self, x):
        out = self.sem_input(x)
        out = self.sem_layers(out)
        out = self.sem_out(out)

        return out

class SemGConv(nn.Module):
    """
    Semantic graph convolution layer
    """

    def __init__(self, adj, in_channels, out_channels, bias=True):
        super(SemGConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.W = nn.Parameter(torch.zeros(size=(2, in_channels, out_channels), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.adj = adj
        self.m = (self.adj > 0)
        self.e = nn.Parameter(torch.zeros(1, len(self.m.nonzero()), dtype=torch.float))
        nn.init.constant_(self.e.data, 1)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        h0 = torch.matmul(input, self.W[0])
        h1 = torch.matmul(input, self.W[1])

        adj = -9e15 * torch.ones_like(self.adj).to(input.device)
        adj[self.m] = self.e
        adj = F.softmax(adj, dim=1)

        M = torch.eye(adj.size(0), dtype=torch.float).to(input.device)
        output = torch.matmul(adj * M, h0) + torch.matmul(adj * (1 - M), h1)

        if self.bias is not None:
            return output + self.bias.view(1, 1, -1)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_channels) + ' -> ' + str(self.out_channels) + ')'
 

class SemGCBlock(nn.Module):
    def __init__(self, adj, in_channels, out_channels):
        super(SemGCBlock, self).__init__()
        self.sem_conv = SemGConv(adj, in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.sem_conv(x).transpose(1, 2)
        x = self.bn(x).transpose(1, 2)
        x = self.relu(x)

        return x


class ResSemGCBlock(nn.Module):
    def __init__(self, adj, in_channels, out_channels, hid_channels):
        super(ResSemGCBlock, self).__init__()

        self.sem_block1 = SemGCBlock(adj, in_channels, hid_channels)
        self.sem_block2 = SemGCBlock(adj, hid_channels, out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        residual = self.shortcut(x) if hasattr(self, 'shortcut') else x
        out = self.sem_block1(x)
        out = self.sem_block2(out)

        return out + residual


class SemGCNet(nn.Module):
    def __init__(self, adj, in_channels, out_channels, hid_channels, num_layers=4):
        super(SemGCNet, self).__init__()

        self.sem_input = SemGCBlock(adj, in_channels, hid_channels)
        
        sem_layers = []
        for _ in range(num_layers):
            sem_layers.append(ResSemGCBlock(adj, hid_channels, hid_channels, hid_channels))

        self.sem_layers = nn.Sequential(*sem_layers)
        self.sem_out = SemGConv(adj, hid_channels, out_channels)

    def forward(self, x):
        out = self.sem_input(x)
        out = self.sem_layers(out)
        out = self.sem_out(out)

        return out

