import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GATConv(nn.Module):
    """
    GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, adj, in_channels, out_channels, alpha=0.2, bias=True):
        super(GATConv, self).__init__()
        self.adj = adj
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.M = torch.eye(adj.size(0), dtype=torch.float)

        self.W = nn.Parameter(torch.zeros(size=(2, in_channels, out_channels)))
        self.a = nn.Parameter(torch.zeros(size=(2*out_channels, 1)))
        self.leakyrelu = nn.LeakyReLU(alpha)

        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        if bias:
            self.bias =  nn.Parameter(torch.zeros(out_channels, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        h0 = torch.matmul(x, self.W[0])
        h1 = torch.matmul(x, self.W[1])
        h = h0 + h1
        B = h.size(0)
        N = h.size(1)

        a_input = torch.cat([h.repeat(1, 1, N).view(B, N * N, -1), h.repeat(1, N, 1)], dim=2).view(B, N, -1, 2 * self.out_channels)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
        adj = self.adj.to(x.device)
        M = self.M.to(x.device)

        zero_vec = -9e15*torch.ones_like(e).to(x.device)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        h_prime = torch.matmul(attention*M, h0) + torch.matmul(attention*(1-M), h1)

        if hasattr(self, 'bias'):
            h_prime += self.bias

        return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_channels) + ' -> ' + str(self.out_channels) + ')'
 

class GATBlock(nn.Module):
    def __init__(self, adj, in_channels, out_channels):
        super(GATBlock, self).__init__()
        self.gat_conv = GATConv(adj, in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.gat_conv(x).transpose(1, 2)
        x = self.bn(x).transpose(1, 2)
        x = self.relu(x)

        return x

class ResGATBlock(nn.Module):
    def __init__(self, adj, in_channels, out_channels, hid_channels):
        super(ResGATBlock, self).__init__()

        self.gat_block1 = GATBlock(adj, in_channels, hid_channels)
        self.gat_block2 = GATBlock(adj, hid_channels, out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        residual = self.shortcut(x) if hasattr(self, 'shortcut') else x
        out = self.gat_block1(x)
        out = self.gat_block2(out)

        return out + residual


class GATNet(nn.Module):
    def __init__(self, adj, in_channels, out_channels, hid_channels, num_layers=4):
        super(GATNet, self).__init__()

        self.gat_input = GATBlock(adj, in_channels, hid_channels)
        
        gat_layers = []
        for _ in range(num_layers):
            gat_layers.append(ResGATBlock(adj, hid_channels, hid_channels, hid_channels))

        self.gat_layers = nn.Sequential(*gat_layers)
        self.gat_out = GATConv(adj, hid_channels, out_channels)

    def forward(self, x):
        out = self.gat_input(x)
        out = self.gat_layers(out)
        out = self.gat_out(out)

        return out

if __name__ == '__main__':
    from graph_utils import adj_matrix_from_num_points
    adj = adj_matrix_from_num_points(98, None, 3)
    gat_net = GATNet(adj, 64, 2, 128, 4)
    x = torch.FloatTensor(32, 98, 64)
    y = gat_net(x)
    print(y.shape)