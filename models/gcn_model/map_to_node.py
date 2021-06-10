import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )

class ATTLayer(nn.Module):
    def __init__(self, channel, reduction=16, with_spatial=True):
        super(ATTLayer, self).__init__()
        self.with_spatial = with_spatial
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        # channel attention
        y1 = self.avg_pool(x).view(b, c)
        y2 = self.max_pool(x).view(b, c)
        y1 = self.fc(y1).view(b, c, 1, 1)
        y2 = self.fc(y2).view(b, c, 1, 1)
        att_c = self.sigmoid(y1+y2)

        if self.with_spatial:
            # spatial attention
            y2 = torch.mean(x,dim=1,keepdim=True)
            y3 = torch.max(x,dim=1,keepdim=True)[0]
            y = torch.cat([y2,y3],1)
            att_s = self.spatial_att(y) # Bx1xWxH

            return att_c , att_s
        else:
            return att_c


class MapToNodeAttention(nn.Module):
    """MapToNodeAttention ......"""
    def __init__(self, in_channels, num_points, expansion_factor=4):
        super(MapToNodeAttention, self).__init__()
        print('expansion_factor is .....{}'.format(expansion_factor))
        _, num_out_feat2, num_out_feat3, num_out_feat4 = in_channels
        self.num_points = num_points

        outfeat2nb = num_out_feat2//2
        self.conv_out2_a = ConvBNReLU(num_out_feat2, outfeat2nb) 
        self.conv_out2 = nn.Sequential(
                            ConvBNReLU(outfeat2nb, outfeat2nb, stride=2),
                            ConvBNReLU(outfeat2nb, outfeat2nb, stride=1),
                            ConvBNReLU(outfeat2nb, outfeat2nb, stride=2))

        outfeat3nb = num_out_feat3//2
        self.conv_out3_a = ConvBNReLU(num_out_feat3, outfeat3nb)
        self.conv_out3 = nn.Sequential(
                            ConvBNReLU(outfeat3nb, outfeat3nb, stride=2),
                            ConvBNReLU(outfeat3nb, outfeat3nb, stride=1))

        outfeat4nb = num_out_feat4//2
        self.conv_out4 = ConvBNReLU(num_out_feat4, outfeat4nb)

        self.conv34 = ConvBNReLU(outfeat4nb+outfeat3nb, outfeat3nb, kernel_size=1)
        self.conv234 = ConvBNReLU(outfeat2nb+outfeat3nb, outfeat2nb, kernel_size=1)

        out_featnb = 256
        self.conv_to_node = nn.Sequential(
                            ConvBNReLU(outfeat2nb+outfeat3nb+outfeat4nb, out_featnb, kernel_size=1),
                            ConvBNReLU(out_featnb, num_points*expansion_factor))
   
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        self.att34 = ATTLayer(outfeat3nb,2)
        self.att234 = ATTLayer(outfeat2nb,2)

    
    def forward(self, x_dict):
        in2 = x_dict['out2']
        in3 = x_dict['out3']
        in4 = x_dict['out4']

        out4 = self.conv_out4(in4)
        out4_up = self.upsample(out4)

        out3 = self.conv_out3_a(in3)
        out34 = torch.cat([out4_up,out3],1)
        out34 = self.conv34(out34)
        out34_att_s, out34_att_c = self.att34(out34)
        out3 = out3 * out34_att_s.expand_as(out3)
        out3 = out3 * out34_att_c.expand_as(out3) + out3
        out3 = self.conv_out3(out3)

        out34_up = self.upsample(out34)
        out2 = self.conv_out2_a(in2)
        out234 = torch.cat([out34_up,out2],1)
        out234 = self.conv234(out234)
        out234_att_s, out234_att_c = self.att234(out234)
        out2 = out2 * out234_att_s.expand_as(out2)
        out2 = out2 * out234_att_c.expand_as(out2) + out2
        out2 = self.conv_out2(out2)
        
        out = torch.cat([out2, out3, out4], 1)
        out = self.conv_to_node(out)
        
        out = out.view(out.size(0), self.num_points, -1) # N x C x (WH)

        return out


class MapToNodeAttentionV2(nn.Module):
    """Same as MapToNodeAttention, but with multi-output: conv-result and reshape-result"""
    def __init__(self, in_channels, num_points, expansion_factor=4):
        super(MapToNodeAttentionV2, self).__init__()
        print('expansion_factor is .....{}'.format(expansion_factor))
        _, num_out_feat2, num_out_feat3, num_out_feat4 = in_channels
        self.num_points = num_points

        outfeat2nb = num_out_feat2//2
        self.conv_out2_a = ConvBNReLU(num_out_feat2, outfeat2nb) 
        self.conv_out2 = nn.Sequential(
                            ConvBNReLU(outfeat2nb, outfeat2nb, stride=2),
                            ConvBNReLU(outfeat2nb, outfeat2nb, stride=1),
                            ConvBNReLU(outfeat2nb, outfeat2nb, stride=2))

        outfeat3nb = num_out_feat3//2
        self.conv_out3_a = ConvBNReLU(num_out_feat3, outfeat3nb)
        self.conv_out3 = nn.Sequential(
                            ConvBNReLU(outfeat3nb, outfeat3nb, stride=2),
                            ConvBNReLU(outfeat3nb, outfeat3nb, stride=1))

        outfeat4nb = num_out_feat4//2
        self.conv_out4 = ConvBNReLU(num_out_feat4, outfeat4nb)

        self.conv34 = ConvBNReLU(outfeat4nb+outfeat3nb, outfeat3nb, kernel_size=1)
        self.conv234 = ConvBNReLU(outfeat2nb+outfeat3nb, outfeat2nb, kernel_size=1)

        # out_featnb = 512 # config_V2
        out_featnb = 256 # config_V22
        self.conv_to_node = nn.Sequential(
                            ConvBNReLU(outfeat2nb+outfeat3nb+outfeat4nb, out_featnb, kernel_size=1),
                            ConvBNReLU(out_featnb, num_points*expansion_factor))
   
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        self.att34 = ATTLayer(outfeat3nb,2)
        self.att234 = ATTLayer(outfeat2nb,2)

    
    def forward(self, x_dict):
        in2 = x_dict['out2']
        in3 = x_dict['out3']
        in4 = x_dict['out4']

        out4 = self.conv_out4(in4)
        out4_up = self.upsample(out4)

        out3 = self.conv_out3_a(in3)
        out34 = torch.cat([out4_up,out3],1)
        out34 = self.conv34(out34)
        out34_att_s, out34_att_c = self.att34(out34)
        out3 = out3 * out34_att_s.expand_as(out3)
        out3 = out3 * out34_att_c.expand_as(out3) + out3
        out3 = self.conv_out3(out3)

        out34_up = self.upsample(out34)
        out2 = self.conv_out2_a(in2)
        out234 = torch.cat([out34_up,out2],1)
        out234 = self.conv234(out234)
        out234_att_s, out234_att_c = self.att234(out234)
        out2 = out2 * out234_att_s.expand_as(out2)
        out2 = out2 * out234_att_c.expand_as(out2) + out2
        out2 = self.conv_out2(out2)
        
        out = torch.cat([out2, out3, out4], 1)
        conv_out = self.conv_to_node(out)
        
        out = conv_out.view(conv_out.size(0), self.num_points, -1) # N x C x (WH)

        return out, conv_out


class MapToNode2b(nn.Module):
    """Simple MapToNode"""
    def __init__(self, in_channels, num_points):
        super(MapToNode2b, self).__init__()
        _, num_out_feat2, num_out_feat3, num_out_feat4 = in_channels
        self.num_points = num_points

        outfeat2nb = num_out_feat2//2
        self.conv_out2_a = nn.Sequential(
            nn.Conv2d(num_out_feat2, outfeat2nb, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(outfeat2nb),
            nn.ReLU(inplace=True))
        self.conv_out2 = nn.Sequential(
            nn.Conv2d(outfeat2nb, outfeat2nb, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(outfeat2nb),
            nn.ReLU(inplace=True),
            nn.Conv2d(outfeat2nb, outfeat2nb, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(outfeat2nb),
            nn.ReLU(inplace=True),
            nn.Conv2d(outfeat2nb, outfeat2nb, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(outfeat2nb),
            nn.ReLU(inplace=True))

        outfeat3nb = num_out_feat3//2
        self.conv_out3_a = nn.Sequential(
            nn.Conv2d(num_out_feat3, outfeat3nb, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(outfeat3nb),
            nn.ReLU(inplace=True))
        self.conv_out3 = nn.Sequential(
            nn.Conv2d(outfeat3nb, outfeat3nb, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(outfeat3nb),
            nn.ReLU(inplace=True),
            nn.Conv2d(outfeat3nb, outfeat3nb, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(outfeat3nb),
            nn.ReLU(inplace=True))


        outfeat4nb = num_out_feat4//2
        self.conv_out4 = nn.Sequential(
            nn.Conv2d(num_out_feat4, outfeat4nb, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(outfeat4nb),
            nn.ReLU(inplace=True))

        self.conv34 = nn.Sequential(
            nn.Conv2d(outfeat4nb+outfeat3nb, outfeat3nb, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(outfeat3nb),
            nn.ReLU(inplace=True))
        self.conv234 = nn.Sequential(
            nn.Conv2d(outfeat2nb+outfeat3nb, outfeat2nb, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(outfeat2nb),
            nn.ReLU(inplace=True))

        out_featnb = 256
        self.conv_to_node = nn.Sequential(
            nn.Conv2d(outfeat2nb+outfeat3nb+outfeat4nb, out_featnb, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(out_featnb),
            nn.ReLU(inplace=True),            
            nn.Conv2d(out_featnb, num_points*4, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_points*4),
            nn.ReLU(inplace=True))
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        self.att34 = ATTLayer(outfeat3nb,2)
        self.att234 = ATTLayer(outfeat2nb,2)

    
    def forward(self, x_dict):
        in2 = x_dict['out2']
        in3 = x_dict['out3']
        in4 = x_dict['out4']

        out4 = self.conv_out4(in4)
        out4_up = self.upsample(out4)

        out3 = self.conv_out3_a(in3)
        out34 = torch.cat([out4_up,out3],1)
        out34 = self.conv34(out34)
        out34_att_s, out34_att_c = self.att34(out34)
        out3 = out3 * out34_att_s.expand_as(out3)
        out3 = out3 * out34_att_c.expand_as(out3) + out3
        out3 = self.conv_out3(out3)

        out34_up = self.upsample(out34)
        out2 = self.conv_out2_a(in2)
        out234 = torch.cat([out34_up,out2],1)
        out234 = self.conv234(out234)
        out234_att_s, out234_att_c = self.att234(out234)
        out2 = out2 * out234_att_s.expand_as(out2)
        out2 = out2 * out234_att_c.expand_as(out2) + out2
        out2 = self.conv_out2(out2)
        
        out = torch.cat([out2, out3, out4], 1)
        out = self.conv_to_node(out)
        
        out = out.view(out.size(0), self.num_points, -1) # N x C x (WH)

        return out


class BaseMapToNode(nn.Module):
    """Base MapToNode, only using feat4"""
    def __init__(self, in_channels, num_points):
        super(BaseMapToNode, self).__init__()
        in_channels = in_channels[-1]
        self.num_points = num_points
        self.conv_out = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))
        self.conv_to_node = nn.Sequential(
            nn.Conv2d(256, num_points*4, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_points*4),
            nn.ReLU(inplace=True))

    def forward(self, x_dict):
        x = x_dict['out4']
        x = self.conv_out(x)
        x = self.conv_to_node(x)
        x = x.view(x.size(0), self.num_points, -1)

        return x


class MapToNode(nn.Module):
    """Simple MapToNode"""
    def __init__(self, in_channels, num_points):
        super(MapToNode, self).__init__()
        _, _, num_out_feat3, num_out_feat4 = in_channels
        self.num_points = num_points
        self.conv_out3 = nn.Sequential(
            nn.Conv2d(num_out_feat3, 128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))

        self.conv_out4 = nn.Sequential(
            nn.Conv2d(num_out_feat4, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))

        self.conv_to_node = nn.Sequential(
            nn.Conv2d(128+128, num_points, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_points),
            nn.ReLU(inplace=True))
    
    def forward(self, x_dict):
        in3 = x_dict['out3']
        in4 = x_dict['out4']

        out3 = self.conv_out3(in3)
        out4 = self.conv_out4(in4)

        out = torch.cat([out3, out4], 1)
        out = self.conv_to_node(out)
        out = out.view(out.size(0), self.num_points, -1) # N x C x (WH)

        return out


