import torch.nn as nn
from models.gcn_model.sem_graph_conv import SemSelfConv
from models.gcn_model.graph_non_local import GraphNonLocal

class _GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, p_dropout=None):
        super(_GraphConv, self).__init__()

        self.gconv = SemSelfConv(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x):
        x = self.gconv(x).transpose(1, 2)
        x = self.bn(x).transpose(1, 2)
        if self.dropout is not None:
            x = self.dropout(self.relu(x))

        x = self.relu(x)
        return x


class _ResGraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim, p_dropout):
        super(_ResGraphConv, self).__init__()

        self.gconv1 = _GraphConv(input_dim, hid_dim, p_dropout)
        self.gconv2 = _GraphConv(hid_dim, output_dim, p_dropout)

    def forward(self, x):
        residual = x
        out = self.gconv1(x)
        out = self.gconv2(out)
        return residual + out


class _GraphNonLocal(nn.Module):
    def __init__(self, hid_dim, grouped_order, restored_order, group_size):
        super(_GraphNonLocal, self).__init__()

        self._nonlocal = GraphNonLocal(hid_dim, sub_sample=group_size)
        self.grouped_order = grouped_order
        self.restored_order = restored_order

    def forward(self, x):
        out = x[:, self.grouped_order, :]
        out = self._nonlocal(out.transpose(1, 2)).transpose(1, 2)
        out = out[:, self.restored_order, :]
        return out


class SemSelfGCN(nn.Module):
    def __init__(self, hid_dim, coords_dim=(2, 2), num_layers=4):
        super(SemSelfGCN, self).__init__()

        _gconv_input = [_GraphConv(coords_dim[0], hid_dim)]
        _gconv_layers = []
        self.num_layers = num_layers
        if num_layers != 0:
            for i in range(num_layers):
                _gconv_layers.append(_ResGraphConv(hid_dim, hid_dim, hid_dim))
            self.gconv_layers = nn.Sequential(*_gconv_layers)
        else:
            print('=========> zeros layers')

        self.gconv_input = nn.Sequential(*_gconv_input)
        self.gconv_output = SemSelfConv(hid_dim, coords_dim[1])

    def forward(self, x):
        out = self.gconv_input(x)
        if self.num_layers != 0:
            out = self.gconv_layers(out)
        out = self.gconv_output(out)
        return out



