import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RelationLearner(nn.Module):
    """RelationLearner
    Returns:
    adjacency matrix 
    """
    def __init__(self, in_channels, num_points, top_k=3):
        super(RelationLearner, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
                nn.Linear(in_channels, in_channels//2),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(in_channels//2),
                nn.Linear(in_channels//2, 470))

    def forward(self, x):
        x = self.avgpool(x).view(x.size(0), -1)
        x = self.fc(x)
        return x