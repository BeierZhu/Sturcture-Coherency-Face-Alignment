# Base Predictor and MultiBranch Predictor
# Date: 2019/09/13
# Author: Beier ZHU

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from utils.log_helper import cprint
from models.gcn_model.map_to_node import MapToNode, MapToNode2b, MapToNodeAttention, MapToNodeAttentionV2, BaseMapToNode
from models.gcn_model.sem_gcn import SemGCN
from models.gcn_model.sem_ch_gcn import SemChGCN
from models.gcn_model.sem_self_gcn import SemSelfGCN
from models.geometric.gatnet import GATNet 
from models.geometric.semgcnet import SemGCNet, MultiHeadSemGCNet
from models.geometric.gcnet import GCNet
from models.geometric.dynamic_gcnet import DynamicGCNet
from models.geometric.dynamic_gcnetV2 import DynamicGCNetV2
from models.geometric.dense_gcn import AdjacencyLearner, DenseGCNet, MultiDenseGCNet
from models.geometric.binary_gcnet import BinaryGCNet
from utils.graph_utils import adj_matrix_from_num_points

import logging
logger = logging.getLogger('FLD')

class MultiDenseGCNPredictor(nn.Module):
    def __init__(self, in_channels, num_points, feat_size, **kwargs):
        super(MultiDenseGCNPredictor, self).__init__()
        logger.debug("build MultiDenseGCNPredictor ......")
        self.map_to_node = MapToNode2b(in_channels=in_channels, num_points=num_points)
        self.adjacency_learner_in = AdjacencyLearner(in_channels=512, feat_size=8, num_points=num_points)
        self.adjacency_learner1 = AdjacencyLearner(in_channels=512, feat_size=8, num_points=num_points)
        self.adjacency_learner2 = AdjacencyLearner(in_channels=512, feat_size=8, num_points=num_points)
        self.adjacency_learner3 = AdjacencyLearner(in_channels=512, feat_size=8, num_points=num_points)
        self.adjacency_learner4 = AdjacencyLearner(in_channels=512, feat_size=8, num_points=num_points)
        self.adjacency_learner_out = AdjacencyLearner(in_channels=512, feat_size=8, num_points=num_points)

        hid_dim = kwargs['hid_dim'] if 'hid_dim' in kwargs else 128
        num_layers = kwargs['num_layers'] if 'num_layers' in kwargs else 4
        self.g_net = MultiDenseGCNet(in_channels=64*4, out_channels=2, hid_channels=hid_dim)

    def forward(self, x_dict):
        adj_in = self.adjacency_learner_in(x_dict['out4'])
        adj1 = self.adjacency_learner1(x_dict['out4'])
        adj2 = self.adjacency_learner2(x_dict['out4'])
        adj3 = self.adjacency_learner3(x_dict['out4'])
        adj4 = self.adjacency_learner4(x_dict['out4'])
        adj_out = self.adjacency_learner_out(x_dict['out4'])
        out = self.map_to_node(x_dict)
        out = self.g_net(out, adj_in, adj1, adj2, adj3, adj4, adj_out)
        out = out.view(out.size(0), -1)
        return out

class DenseGCNPredictor(nn.Module):
    def __init__(self, in_channels, num_points, feat_size, **kwargs):
        super(DenseGCNPredictor, self).__init__()
        logger.debug("build DenseGCNPredictor ......")
        self.map_to_node = MapToNode2b(in_channels=in_channels, num_points=num_points)
        self.adjacency_learner =AdjacencyLearner(in_channels=512, feat_size=8, num_points=num_points)
        hid_dim = kwargs['hid_dim'] if 'hid_dim' in kwargs else 128
        num_layers = kwargs['num_layers'] if 'num_layers' in kwargs else 4
        self.g_net = DenseGCNet(in_channels=64*4, out_channels=2, hid_channels=hid_dim, num_layers=num_layers)

    def forward(self, x_dict):
        adj = self.adjacency_learner(x_dict['out4'])
        out = self.map_to_node(x_dict)
        out = self.g_net(out, adj)
        out = out.view(out.size(0), -1)
        return out


class LightSemGraphPredictorV2b(nn.Module):
    def __init__(self, in_channels, num_points, feat_size, **kwargs):
        super(LightSemGraphPredictorV2b, self).__init__()
        cprint.green("Creating LightSemGraphPredictorV2b ......")
        _, _, num_out_feat3, num_out_feat4 = in_channels  
        self.map_to_node = MapToNode2b(in_channels=in_channels, num_points=num_points)
        top_k = kwargs['top_k'] if 'top_k' in kwargs else False
        adj_matrix = adj_matrix_from_num_points(num_points=num_points, is_weight=False, top_k=top_k)
        hid_dim = kwargs['hid_dim'] if 'hid_dim' in kwargs else 64
        num_layers = kwargs['num_layers'] if 'num_layers' in kwargs else 2
        p_dropout = kwargs['p_dropout'] if 'p_dropout' in kwargs else None
        self.sem_gcn = SemGCN(adj_matrix, hid_dim=hid_dim, num_layers=num_layers, coords_dim=(64*4, 2), p_dropout=p_dropout)


    def forward(self, x_dict):
        out = self.map_to_node(x_dict)
        out = self.sem_gcn(out)
        out = out.view(out.size(0), -1)
        return out
 

class GCNetPredictorV2b(nn.Module):
    def __init__(self, in_channels, num_points, feat_size, **kwargs):
        super(GCNetPredictorV2b, self).__init__()
        logger.debug("build GCNetPredictorV2b ......")
        self.map_to_node = MapToNode2b(in_channels=in_channels, num_points=num_points)
        top_k = kwargs['top_k'] if 'top_k' in kwargs else False
        is_weight = kwargs['is_weight'] if 'is_weight' in kwargs else False
        adj_matrix = adj_matrix_from_num_points(num_points=num_points, is_weight=is_weight, top_k=top_k)
        hid_dim = kwargs['hid_dim'] if 'hid_dim' in kwargs else 128
        num_layers = kwargs['num_layers'] if 'num_layers' in kwargs else 4
        p_dropout = kwargs['p_dropout'] if 'p_dropout' in kwargs else None
        self.g_net = GCNet(adj_matrix, in_channels=64*4, out_channels=2, hid_channels=hid_dim, num_layers=num_layers)

    def forward(self, x_dict):
        out = self.map_to_node(x_dict)
        out = self.g_net(out)
        out = out.view(out.size(0), -1)
        return out


class BinaryGCNetPredictor(nn.Module):
    def __init__(self, in_channels, num_points, feat_size, **kwargs):
        super(BinaryGCNetPredictor, self).__init__()
        logger.debug("build BinaryGCNetPredictor ......")
        self.map_to_node = MapToNode2b(in_channels=in_channels, num_points=num_points)
        top_k = kwargs['top_k'] if 'top_k' in kwargs else False
        is_weight = kwargs['is_weight'] if 'is_weight' in kwargs else False
        adj_matrix = adj_matrix_from_num_points(num_points=num_points, is_weight=is_weight, top_k=top_k)
        hid_dim = kwargs['hid_dim'] if 'hid_dim' in kwargs else 128
        num_layers = kwargs['num_layers'] if 'num_layers' in kwargs else 4
        p_dropout = kwargs['p_dropout'] if 'p_dropout' in kwargs else None
        self.g_net = BinaryGCNet(adj_matrix, in_channels=64*4, out_channels=2, hid_channels=hid_dim, num_layers=num_layers)

    def forward(self, x_dict):
        out = self.map_to_node(x_dict)
        out = self.g_net(out)
        out = out.view(out.size(0), -1)
        return out


class MultiHeadSemGraphPredictorV2b(nn.Module):
    def __init__(self, in_channels, num_points, feat_size, **kwargs):
        super(MultiHeadSemGraphPredictorV2b, self).__init__()
        cprint.green("Creating MultiHeadSemGraphPredictorV2b ......")
        _, _, num_out_feat3, num_out_feat4 = in_channels  
        self.map_to_node = MapToNode2b(in_channels=in_channels, num_points=num_points)
        top_k = kwargs['top_k'] if 'top_k' in kwargs else False
        adj_matrix = adj_matrix_from_num_points(num_points=num_points, is_weight=False, top_k=top_k)
        hid_dim = kwargs['hid_dim'] if 'hid_dim' in kwargs else 128
        num_layers = kwargs['num_layers'] if 'num_layers' in kwargs else 4
        num_head = kwargs['num_head'] if 'num_head' in kwargs else None
        self.sem_gcn = MultiHeadSemGCNet(adj_matrix, in_channels=64*4, out_channels=2,hid_channels=hid_dim, num_layers=num_layers, num_head=num_head)

    def forward(self, x_dict):
        out = self.map_to_node(x_dict)
        out = self.sem_gcn(out)
        out = out.view(out.size(0), -1)
        return out


class StructureCoherencePredictor(nn.Module):
    def __init__(self, in_channels, num_points, feat_size, **kwargs):
        super(StructureCoherencePredictor, self).__init__()
        cprint.green("Creating StructureCoherencePredictor ......")
        _, _, num_out_feat3, num_out_feat4 = in_channels  
        top_k = kwargs['top_k'] if 'top_k' in kwargs else False
        expansion_factor = kwargs['expansion_factor'] if 'expansion_factor' in kwargs else 4
        self.map_to_node = MapToNodeAttention(in_channels=in_channels, num_points=num_points, expansion_factor=expansion_factor)
        adj_matrix = adj_matrix_from_num_points(num_points=num_points, is_weight=False, top_k=top_k)
        hid_dim = kwargs['hid_dim'] if 'hid_dim' in kwargs else 128
        num_layers = kwargs['num_layers'] if 'num_layers' in kwargs else 4
        p_dropout = kwargs['p_dropout'] if 'p_dropout' in kwargs else None
        self.sem_gcn = SemGCN(adj_matrix, hid_dim=hid_dim, num_layers=num_layers, coords_dim=(64*expansion_factor, 2), p_dropout=p_dropout)


    def forward(self, x_dict):
        out = self.map_to_node(x_dict)
        out = self.sem_gcn(out)
        out = out.view(out.size(0), -1)
        return out


class SemGraphPredictorV2b(nn.Module):
    def __init__(self, in_channels, num_points, feat_size, **kwargs):
        super(SemGraphPredictorV2b, self).__init__()
        cprint.green("Creating SemGraphPredictorV2b ......")
        _, _, num_out_feat3, num_out_feat4 = in_channels  
        self.map_to_node = MapToNode2b(in_channels=in_channels, num_points=num_points)
        top_k = kwargs['top_k'] if 'top_k' in kwargs else False
        adj_matrix = adj_matrix_from_num_points(num_points=num_points, is_weight=False, top_k=top_k)
        hid_dim = kwargs['hid_dim'] if 'hid_dim' in kwargs else 128
        num_layers = kwargs['num_layers'] if 'num_layers' in kwargs else 4
        p_dropout = kwargs['p_dropout'] if 'p_dropout' in kwargs else None
        self.sem_gcn = SemGCN(adj_matrix, hid_dim=hid_dim, num_layers=num_layers, coords_dim=(feat_size*feat_size*4, 2), p_dropout=p_dropout)


    def forward(self, x_dict):
        out = self.map_to_node(x_dict)
        out = self.sem_gcn(out)
        out = out.view(out.size(0), -1)
        return out


class BaseGCNPredictor(nn.Module):
    def __init__(self, in_channels, num_points, feat_size, **kwargs):
        super(BaseGCNPredictor, self).__init__()
        cprint.green("Creating BaseGCNPredictor ......")
        _, _, num_out_feat3, num_out_feat4 = in_channels  
        self.map_to_node = BaseMapToNode(in_channels=in_channels, num_points=num_points)
        top_k = kwargs['top_k'] if 'top_k' in kwargs else False
        adj_matrix = adj_matrix_from_num_points(num_points=num_points, is_weight=False, top_k=top_k)
        hid_dim = kwargs['hid_dim'] if 'hid_dim' in kwargs else 128
        num_layers = kwargs['num_layers'] if 'num_layers' in kwargs else 4
        p_dropout = kwargs['p_dropout'] if 'p_dropout' in kwargs else None
        self.sem_gcn = SemGCN(adj_matrix, hid_dim=hid_dim, num_layers=num_layers, coords_dim=(feat_size*feat_size*4, 2), p_dropout=p_dropout)

    def forward(self, x_dict):
        out = self.map_to_node(x_dict)
        out = self.sem_gcn(out)
        out = out.view(out.size(0), -1)
        return out


class FCPredictor(nn.Module):
    def __init__(self, in_channels, num_points, feat_size, **kwargs):
        super(FCPredictor, self).__init__()
        cprint.green("Creating FCPredictor ......")
        _, _, num_out_feat3, num_out_feat4 = in_channels  
        self.fc1 = nn.Linear(8*8*num_out_feat4, 256)
        self.fc2 = nn.Linear(256, num_points*2)
        self.relu = nn.ReLU()        


    def forward(self, x_dict):
        out = x_dict['out4']
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
    
        return out


class AttentionFCPredictor(nn.Module):
    def __init__(self, in_channels, num_points, feat_size, **kwargs):
        super(AttentionFCPredictor, self).__init__()
        cprint.green("Creating AttentionFCPredictor ......")
        _, _, num_out_feat3, num_out_feat4 = in_channels  
        self.map_to_node = MapToNode2b(in_channels=in_channels, num_points=num_points)
        self.fc = nn.Linear(num_points*feat_size*feat_size*4, num_points*2)

    def forward(self, x_dict):
        out = self.map_to_node(x_dict)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


class AdapAvgPoolPredictor(nn.Module):
    def __init__(self, in_channels, feat_size, num_points):
        super(AdapAvgPoolPredictor, self).__init__()
        cprint.green("Creating AdapAvgPoolPredictor ......")
        self.glob_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.PReLU(num_parameters=in_channels),
            nn.Linear(in_channels, num_points*2))

    def forward(self, x):
        x = self.glob_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
        

class MultiFeaturePredictor(nn.Module):
    """
    Build multi-feature model from cfg
    """

    def __init__(self, in_channels, feat_size, num_points, **kwargs):
        super(MultiFeaturePredictor, self).__init__()
        cprint.green("Creating MultiFeaturePredictor ......")
        self.avg_glob_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_1 = nn.Conv2d(in_channels[-1], 32, kernel_size=3, padding=1, stride=2) # before: feat_size, after: feat_size//2
        self.conv_2 = nn.Conv2d(32, 128, kernel_size=feat_size//2) 
        self.relu = nn.ReLU(inplace=True)

        self.fc = nn.Linear(in_channels[-1]+32+128, num_points*2)

    def forward(self, x_dict):
        x = x_dict['out4']
        out1 = self.avg_glob_pool(x)
        out1 = out1.view(out1.size(0), -1)

        x = self.conv_1(x)
        x = self.relu(x)

        out2 = self.avg_glob_pool(x)
        out2 = out2.view(out2.size(0), -1)

        out3 = self.conv_2(x)
        out3 = out3.view(out3.size(0), -1)

        out = torch.cat([out1, out2, out3], 1)
        out = self.fc(out)

        return out

class MultiFeaturePredictorV2(nn.Module):
    def __init__(self, in_channels, feat_size, num_points,**kwargs):
        super(MultiFeaturePredictorV2, self).__init__()
        _, _, num_out_feat3, num_out_feat4 = in_channels  
        self.avg_glob_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_out3 = nn.Sequential(
            nn.Conv2d(num_out_feat3, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        self.conv_out4 = nn.Sequential(
            nn.Conv2d(num_out_feat4, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        self.conv_global_pool = nn.Conv2d(num_out_feat4, 128, kernel_size=feat_size)
        self.fc = nn.Linear(128+128+128, num_points*2)


    def forward(self, x_dict):
        in3 = x_dict['out3']
        in4 = x_dict['out4']

        out3 = self.conv_out3(in3)
        out3 = self.avg_glob_pool(out3)
        out3 = out3.view(out3.size(0), -1)

        out4 = self.conv_out4(in4)
        out4 = self.avg_glob_pool(out4)
        out4 = out4.view(out4.size(0), -1)

        out5 = self.conv_global_pool(in4)
        out5 = out5.view(out5.size(0), -1)

        out = torch.cat([out3, out4, out5], 1)
        out = self.fc(out)

        return out


class _SemGraphPredictor(nn.Module):
    """docstring for _SemGraphPredictor"""
    def __init__(self, map_to_node, in_channels, feat_size, num_points, **kwargs):
        super(_SemGraphPredictor, self).__init__()
        self.map_to_node = map_to_node(in_channels=in_channels, num_points=num_points)
        adj_matrix = adj_matrix_from_num_points(num_points=num_points, is_weight=False)
        hid_dim = kwargs['hid_dim'] if 'hid_dim' in kwargs else 128
        num_layers = kwargs['num_layers'] if 'num_layers' in kwargs else 4
        p_dropout = kwargs['p_dropout'] if 'p_dropout' in kwargs else None
        self.sem_gcn = SemGCN(adj_matrix, hid_dim=hid_dim, num_layers=num_layers, coords_dim=(feat_size**2, 2), p_dropout=p_dropout)

    def forward(self, x_dict):
        out = self.map_to_node(x_dict)
        out = self.sem_gcn(out)
        out = out.view(out.size(0), -1)
        return out

class _SemChGraphPredictor(nn.Module):
    """docstring for _SemGraphPredictor"""
    def __init__(self, map_to_node, in_channels, feat_size, num_points, **kwargs):
        super(_SemChGraphPredictor, self).__init__()
        self.map_to_node = map_to_node(in_channels=in_channels, num_points=num_points)
        adj_matrix = adj_matrix_from_num_points(num_points=num_points, is_weight=False)
        hid_dim = kwargs['hid_dim'] if 'hid_dim' in kwargs else 128
        num_layers = kwargs['num_layers'] if 'num_layers' in kwargs else 4
        p_dropout = kwargs['p_dropout'] if 'p_dropout' in kwargs else None
        self.sem_gcn = SemChGCN(adj_matrix, hid_dim=hid_dim, num_layers=num_layers, coords_dim=(feat_size**2, 2), p_dropout=p_dropout)

    def forward(self, x_dict):
        out = self.map_to_node(x_dict)
        out = self.sem_gcn(out)
        out = out.contiguous().view(out.size(0), -1)
        return out

def SemChGraphPredictor(in_channels, num_points, feat_size, **kwargs):
    cprint.green('Creating SemChGraphPredictor')
    return _SemChGraphPredictor(map_to_node=MapToNode, in_channels=in_channels,
        feat_size=feat_size, num_points=num_points)


class GCNetPredictor(nn.Module):
    def __init__(self, in_channels, num_points, feat_size, **kwargs):
        super(GCNetPredictor, self).__init__()
        logger.debug("build GCNetPredictor ......")
        self.map_to_node = MapToNode(in_channels=in_channels, num_points=num_points)
        top_k = kwargs['top_k'] if 'top_k' in kwargs else False
        is_weight = kwargs['is_weight'] if 'is_weight' in kwargs else False
        adj_matrix = adj_matrix_from_num_points(num_points=num_points, is_weight=is_weight, top_k=top_k)
        hid_dim = kwargs['hid_dim'] if 'hid_dim' in kwargs else 128
        num_layers = kwargs['num_layers'] if 'num_layers' in kwargs else 4
        p_dropout = kwargs['p_dropout'] if 'p_dropout' in kwargs else None
        self.g_net = GCNet(adj_matrix, in_channels=feat_size**2, out_channels=2, hid_channels=hid_dim, num_layers=num_layers)

    def forward(self, x_dict):
        out = self.map_to_node(x_dict)
        out = self.g_net(out)
        out = out.view(out.size(0), -1)
        return out
        
class DynamicGCNetPredictor(nn.Module):
    """docstring for DynamicGCNetPredictor"""
    def __init__(self, in_channels, num_points, feat_size, **kwargs):
        super(DynamicGCNetPredictor, self).__init__()
        logger.debug("build DynamicGCNetPredictor ......")
        self.map_to_node = BaseMapToNode(in_channels=in_channels, num_points=num_points)
        top_k = kwargs['top_k'] if 'top_k' in kwargs else False
        is_weight = kwargs['is_weight'] if 'is_weight' in kwargs else False
        adj_matrix = adj_matrix_from_num_points(num_points=num_points, is_weight=is_weight, top_k=top_k)
        hid_dim = kwargs['hid_dim'] if 'hid_dim' in kwargs else 128
        num_layers = kwargs['num_layers'] if 'num_layers' in kwargs else 4
        self.g_net = DynamicGCNet(adj_matrix, in_channels=feat_size**2*4, out_channels=2, num_points=num_points, hid_channels=hid_dim, top_k=top_k, num_layers=num_layers)

    def forward(self, x_dict):
        out4 = x_dict['out4']
        out = self.map_to_node(x_dict)
        out = self.g_net(out4, out)
        out = out.view(out.size(0), -1)
        return out

class DynamicGCNetPredictorV2(nn.Module):
    """docstring for DynamicGCNetPredictor"""
    def __init__(self, in_channels, num_points, feat_size, **kwargs):
        super(DynamicGCNetPredictorV2, self).__init__()
        logger.debug("build DynamicGCNetPredictorV2 ......")
        self.map_to_node = MapToNodeAttentionV2(in_channels=in_channels, num_points=num_points)
        top_k = kwargs['top_k'] if 'top_k' in kwargs else False
        is_weight = kwargs['is_weight'] if 'is_weight' in kwargs else False
        adj_matrix = adj_matrix_from_num_points(num_points=num_points, is_weight=is_weight, top_k=top_k)
        hid_dim = kwargs['hid_dim'] if 'hid_dim' in kwargs else 128
        num_layers = kwargs['num_layers'] if 'num_layers' in kwargs else 4
        self.g_net = DynamicGCNetV2(adj_matrix, in_channels=feat_size**2*4, out_channels=2, num_points=num_points, hid_channels=hid_dim, top_k=top_k, num_layers=num_layers)

    def forward(self, x_dict):
        out, conv_out = self.map_to_node(x_dict)
        out = self.g_net(conv_out, out)
        out = out.view(out.size(0), -1)
        return out


class SelfPredictor(nn.Module):
    """docstring for DynamicGCNetPredictor"""
    def __init__(self, in_channels, num_points, feat_size, **kwargs):
        super(SelfPredictor, self).__init__()
        logger.debug("build SelfPredictor ......")
        self.map_to_node = MapToNode2b(in_channels=in_channels, num_points=num_points)
        hid_dim = kwargs['hid_dim'] if 'hid_dim' in kwargs else 128
        num_layers = kwargs['num_layers'] if 'num_layers' in kwargs else 4
        self.g_net = SemSelfGCN(hid_dim=hid_dim, coords_dim=(feat_size**2*4, 2), num_layers=num_layers)

    def forward(self, x_dict):
        out = self.map_to_node(x_dict)
        out = self.g_net(out)
        out = out.view(out.size(0), -1)
        return out

class SemGCNetPredictor(nn.Module):
    def __init__(self, in_channels, num_points, feat_size, **kwargs):
        super(SemGCNetPredictor, self).__init__()
        logger.debug("build SemGCNetPredictor ......")
        self.map_to_node = MapToNode(in_channels=in_channels, num_points=num_points)
        top_k = kwargs['top_k'] if 'top_k' in kwargs else False
        is_weight = kwargs['is_weight'] if 'is_weight' in kwargs else False
        adj_matrix = adj_matrix_from_num_points(num_points=num_points, is_weight=is_weight, top_k=top_k)
        hid_dim = kwargs['hid_dim'] if 'hid_dim' in kwargs else 128
        num_layers = kwargs['num_layers'] if 'num_layers' in kwargs else 4
        p_dropout = kwargs['p_dropout'] if 'p_dropout' in kwargs else None
        self.sem_net = SemGCNet(adj_matrix, in_channels=feat_size**2, out_channels=2, hid_channels=hid_dim, num_layers=num_layers)

    def forward(self, x_dict):
        out = self.map_to_node(x_dict)
        out = self.sem_net(out)
        out = out.view(out.size(0), -1)
        return out


class GATPredictor(nn.Module):
    def __init__(self, in_channels, num_points, feat_size, **kwargs):
        super(GATPredictor, self).__init__()
        logger.debug("build GATPredictor ......")
        self.map_to_node = MapToNode(in_channels=in_channels, num_points=num_points)
        top_k = kwargs['top_k'] if 'top_k' in kwargs else False
        adj_matrix = adj_matrix_from_num_points(num_points=num_points, is_weight=False, top_k=top_k)
        hid_dim = kwargs['hid_dim'] if 'hid_dim' in kwargs else 128
        num_layers = kwargs['num_layers'] if 'num_layers' in kwargs else 4
        p_dropout = kwargs['p_dropout'] if 'p_dropout' in kwargs else None
        self.gat_net = GATNet(adj_matrix, in_channels=feat_size**2, out_channels=2, hid_channels=hid_dim, num_layers=num_layers)

    def forward(self, x_dict):
        out = self.map_to_node(x_dict)
        out = self.gat_net(out)
        out = out.view(out.size(0), -1)
        return out


class SemGraphPredictor(nn.Module):
    def __init__(self, in_channels, num_points, feat_size, **kwargs):
        super(SemGraphPredictor, self).__init__()
        logger.debug("creating SemGraphPredictor ......")
        self.map_to_node = MapToNode(in_channels=in_channels, num_points=num_points)
        top_k = kwargs['top_k'] if 'top_k' in kwargs else False
        adj_matrix = adj_matrix_from_num_points(num_points=num_points, is_weight=False, top_k=top_k)
        hid_dim = kwargs['hid_dim'] if 'hid_dim' in kwargs else 128
        num_layers = kwargs['num_layers'] if 'num_layers' in kwargs else 4
        p_dropout = kwargs['p_dropout'] if 'p_dropout' in kwargs else None
        self.sem_gcn = SemGCN(adj_matrix, hid_dim=hid_dim, num_layers=num_layers, coords_dim=(feat_size**2, 2), p_dropout=p_dropout)

    def forward(self, x_dict):
        out = self.map_to_node(x_dict)
        out = self.sem_gcn(out)
        out = out.view(out.size(0), -1)
        return out    



