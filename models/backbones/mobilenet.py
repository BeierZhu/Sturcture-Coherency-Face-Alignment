from torch import nn


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self,
                 width_mult=1.0,
                 is_color=True,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet

        """
        super(MobileNetV2, self).__init__()

        if block is None:
            block = InvertedResidual
        
        self.num_out_feats = [inverted_residual_setting['setting1'][0][1], 
                              inverted_residual_setting['setting2'][0][1],
                              inverted_residual_setting['setting3'][0][1],
                              inverted_residual_setting['setting4'][0][1]]
        self.downsample_ratio = 32

        self.input_channel = 32
        self.conv_bn_relu1 = ConvBNReLU(3 if is_color else 1, self.input_channel, stride=2)
        
        self.layer0 = self._make_layer(block, inverted_residual_setting['setting0'])
        self.layer1 = self._make_layer(block, inverted_residual_setting['setting1'])
        self.layer2 = self._make_layer(block, inverted_residual_setting['setting2'])
        self.layer3 = self._make_layer(block, inverted_residual_setting['setting3'])
        self.layer4 = self._make_layer(block, inverted_residual_setting['setting4'])

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _make_layer(self, block, setting):
        layers = []
        for t, c, n, s in setting:
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(block(self.input_channel, c, stride, expand_ratio=t))
                self.input_channel = c
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_bn_relu1(x)
        x = self.layer0(x)
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        x_dict = {'out1': out1, 'out2': out2, 'out3': out3, 'out4':out4}
        return x_dict

def MobileNetA(is_color=True,**kwargs):
    print('create mobilenetA')

    inverted_residual_setting={
    'setting0': [[1, 16, 1, 1],
                 [6, 24, 2, 2],
                 [6, 32, 3, 2]],
    'setting1': [[6, 64, 4, 2]],
    'setting2': [[6, 96, 3, 1]],
    'setting3': [[6, 160, 3, 2]],
    'setting4': [[6, 320, 1, 1]]
    }
        
    model = MobileNetV2(is_color=is_color, inverted_residual_setting=inverted_residual_setting)
    return model

def MobileNetB(is_color=True,**kwargs):
    print('create mobilenetB')

    inverted_residual_setting={
    'setting0': [[6, 32, 2, 2]],
    'setting1': [[6, 64, 4, 1]],
    'setting2': [[6, 128, 3, 2]],
    'setting3': [[6, 256, 3, 2]],
    'setting4': [[6, 512, 1, 2]] 
    }
        
    model = MobileNetV2(is_color=is_color, inverted_residual_setting=inverted_residual_setting)
    return model

def MobileNetC(is_color=True,**kwargs):
    print('create mobilenetC')

    inverted_residual_setting={
    'setting0': [[6, 16, 2, 2]],
    'setting1': [[6, 32, 4, 1]],
    'setting2': [[6, 64, 3, 2]],
    'setting3': [[6, 128, 3, 2]],
    'setting4': [[6, 256, 1, 2]] 
    }
        
    model = MobileNetV2(is_color=is_color, inverted_residual_setting=inverted_residual_setting)
    return model


if __name__ == '__main__':
    import torch
    model = MobileNetV2()
    x = torch.Tensor(1, 3, 256, 256)
    x_dict = model(x)
    print(x_dict['out1'].shape)
    print(x_dict['out2'].shape)
    print(x_dict['out3'].shape)
    print(x_dict['out4'].shape)
