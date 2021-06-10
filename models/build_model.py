# Model Builder
# Date: 2019/09/13
# Author: Beier ZHU
import torch.nn as nn

from models import predictor 
from models.backbones import resnet, resnet_pure
from models.backbones import preact_resnet 
from models.backbones import preact_resnet_ibn
from models.backbones import preact_resnet_coordconv_ibn
from models.backbones import preact_resnet_antialias
from models.backbones import mobilenet
from models.gcn_model import sem_gcn
from models.backbones import hrnet
from models.backbones import efficient_net
from models.backbones import vgg


backbone_zoo = {
    'hrnetw32': hrnet.hrnetw32,
    'hrnet': hrnet.get_face_alignment_net,
    'efficientnet_b0': efficient_net.efficientnet_b0,
    'efficientnet_b1': efficient_net.efficientnet_b1,
    'efficientnet_b0_pretrained': efficient_net.efficientnet_b0_pretrained,
    'efficientnet_b1_pretrained': efficient_net.efficientnet_b1_pretrained,
    'ResNet18': resnet.ResNet18,
    'ResNet34': resnet.ResNet34,
    'ResNet18Pure': resnet_pure.ResNet18,
    'ResNet34Pure': resnet_pure.ResNet34,
    'PreActResNet18': preact_resnet.PreActResNet18,
    'PreActResNet18IBN': preact_resnet_ibn.PreActResNet18IBN,
    'PreActResNet18IBN_a': preact_resnet_ibn.PreActResNet18IBN_a,
    'PreActResNet18IBN_b': preact_resnet_ibn.PreActResNet18IBN_b,
    'PreActResNet34': preact_resnet.PreActResNet34,
    'PreActResNet34IBN': preact_resnet_ibn.PreActResNet34IBN,
    'PreActResNet34IBN_a': preact_resnet_ibn.PreActResNet34IBN_a,
    'PreActResNet34IBN_b': preact_resnet_ibn.PreActResNet34IBN_b,
    'PreActResNet50IBN_a': preact_resnet_ibn.PreActResNet50IBN_a,
    'PreActResNet50': preact_resnet.PreActResNet50,
    'PreActResNet101': preact_resnet.PreActResNet101,
    'PreActResNet152': preact_resnet.PreActResNet152,
    'LightPreActResNet18IBN_a': preact_resnet_ibn.LightPreActResNet18IBN_a,
    'LightPreActResNet34IBN_a': preact_resnet_ibn.LightPreActResNet34IBN_a,
    'MobileNetA': mobilenet.MobileNetA,
    'MobileNetB': mobilenet.MobileNetB,
    'MobileNetC': mobilenet.MobileNetC,
    'PreActResNet34CoordConv': preact_resnet_coordconv_ibn.PreActResNet34CoordConv,
    'PreActResNetAntiAliased34': preact_resnet_antialias.PreActResNetAntiAliased34,
    'Vgg16': vgg.Vgg16,
}

predictor_zoo = {
    'AdapAvgPoolPredictor': predictor.AdapAvgPoolPredictor,
    'MultiFeaturePredictor': predictor.MultiFeaturePredictor,
    'MultiFeaturePredictorV2': predictor.MultiFeaturePredictorV2, 
    'SemGraphPredictor': predictor.SemGraphPredictor,
    'SemChGraphPredictor': predictor.SemChGraphPredictor,
    'GATPredictor': predictor.GATPredictor,
    'SemGCNetPredictor': predictor.SemGCNetPredictor,
    'GCNetPredictor': predictor.GCNetPredictor,
    'DynamicGCNetPredictor': predictor.DynamicGCNetPredictor,
    'DynamicGCNetPredictorV2': predictor.DynamicGCNetPredictorV2,
    'SemGraphPredictorV2b': predictor.SemGraphPredictorV2b,
    'GCNetPredictorV2b': predictor.GCNetPredictorV2b,
    'DenseGCNPredictor': predictor.DenseGCNPredictor,
    'MultiDenseGCNPredictor': predictor.MultiDenseGCNPredictor,
    'LightSemGraphPredictorV2b': predictor.LightSemGraphPredictorV2b,
    'MultiHeadSemGraphPredictorV2b': predictor.MultiHeadSemGraphPredictorV2b,
    'StructureCoherencePredictor': predictor.StructureCoherencePredictor,
    'AttentionFCPredictor': predictor.AttentionFCPredictor,
    'SelfPredictor': predictor.SelfPredictor,
    'BinaryGCNetPredictor': predictor.BinaryGCNetPredictor,
    'BaseGCNPredictor': predictor.BaseGCNPredictor,
    'FCPredictor': predictor.FCPredictor
}

class ModelBuilder(nn.Module):
    """
    Build model from cfg
    """
    def __init__(self, config):
        super(ModelBuilder, self).__init__()
        is_color = config.is_color if hasattr(config, 'is_color') else True
        pretrained_path = config.pretrained_path if hasattr(config, 'pretrained_path') else None
        receptive_keep = True if hasattr(config, 'receptive_keep') and config.receptive_keep else False
        
        self.backbone = backbone_zoo[config.backbone_name](is_color=is_color, 
                                                           pretrained_path=pretrained_path, 
                                                           receptive_keep=receptive_keep)
        # in_channels = self.backbone.num_out_feats[-1]
        # feat_size = int(math.ceil(config.crop_size/self.backbone.downsample_ratio))
        feat_size = config.crop_size//self.backbone.downsample_ratio
        gcn_param = config.gcn_param if hasattr(config, 'gcn_param') else {}
        self.predictor = predictor_zoo[config.predictor_name](in_channels=self.backbone.num_out_feats, 
                                                              feat_size=feat_size, 
                                                              num_points=config.num_kpts,
                                                              **gcn_param)

    def forward(self, x):
        x = self.backbone(x)
        x = self.predictor(x)

        return x


