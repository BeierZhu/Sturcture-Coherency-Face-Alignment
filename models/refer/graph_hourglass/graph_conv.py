import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConv(nn.Module):
    """Simple GCN layer"""
    def __init__(self, in_features, out_features, adj, bias=True):
        super(GraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.adj = adj
        self.W = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.W.data, a=0, mode='fan_in')
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    def forward(self, x):
        support = torch.mm(x, self.W)
        output = torch.spmm(self.adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class LayoutGraphConv(object):
    """LayoutGraphConv used in 'Layout-Graph Reasoning for Fasion Landmark Detection' """
    def __init__(self, arg):
        super(LayoutGraphConv, self).__init__()
        pass
        
     
class SemGraphConv(object):
    """Simple GCN layer with trainable weight on adjcent matrix"""
    def __init__(self, in_features, out_features, adj, bias=True):
        super(SemGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.adj = adj
        self.m = self.adj > 0
        self.W = nn.Parameter(torch.FloatTensor(2, in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))       
        else:
            self.register_parameter('bias', None)

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        stdv = 1. / math.sqrt(self.W.size(2))
        self.bias.data.uniform_(-stdv, stdv)



        