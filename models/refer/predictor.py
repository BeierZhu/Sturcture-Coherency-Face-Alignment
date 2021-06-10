"""
branch_index_68pts = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],   # contour                0
    [17, 18, 19, 20, 21],                                         # left top eyebrow       1
    [22, 23, 24, 25, 26],                                         # right top eyebrow      2
    [27, 28, 29, 30, 31, 32, 33, 34, 35],                         # nose bridge & tip      3
    [36, 37, 38, 39, 40, 41],                                     # left top & bottom eye  4
    [42, 43, 44, 45, 46, 47],                                     # right top & bottom eye 5
    [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, \
     58, 59, 60, 61, 62, 63, 64, 65, 66, 67]                      # lip                    6                                    
]
"""

class MultiBranchPredictor(nn.Module):
    def __init__(self, in_channels, feat_size, num_points, **kwargs):
        super(MultiBranchPredictor, self).__init__()
        cprint.green("Creating MultiBranchPredictor ......")
        in_channels = in_channels[-1]

        self.conv_contour = nn.Conv2d(in_channels, in_channels, kernel_size=feat_size)         # contour
        self.conv_left_eyebrow = nn.Conv2d(in_channels, in_channels, kernel_size=feat_size)    # left top eyebrow + left bottom eyebrow (98pt)
        self.conv_right_eyebrow = nn.Conv2d(in_channels, in_channels, kernel_size=feat_size)   # right top eyebrow + right bottom eyebrow (98pt)
        self.conv_nose = nn.Conv2d(in_channels, in_channels, kernel_size=feat_size)            # nose bridge + nose tip
        self.conv_left_eye = nn.Conv2d(in_channels, in_channels, kernel_size=feat_size)        # left top eye + left bottom eye
        self.conv_right_eye = nn.Conv2d(in_channels, in_channels, kernel_size=feat_size)       # right top eye + right bottom eye
        self.conv_lip = nn.Conv2d(in_channels, in_channels, kernel_size=feat_size)             # up up lip + up bottom lip + bottom up lip + bottom bottom lip
        self.conv_left_eyebrow_b = nn.Conv2d(in_channels, in_channels, kernel_size=feat_size)
        self.conv_right_eyebrow_b = nn.Conv2d(in_channels, in_channels, kernel_size=feat_size)

       
        self.classifier_contour = nn.Linear(in_channels, 33*2)
        self.classifier_left_eyebrow = nn.Linear(in_channels, 5*2)
        self.classifier_right_eyebrow = nn.Linear(in_channels, 5*2)
        self.classifier_nose = nn.Linear(in_channels, 9*2)
        self.classifier_left_eye = nn.Linear(in_channels, 8*2)
        self.classifier_right_eye = nn.Linear(in_channels, 8*2)
        self.classifier_lip = nn.Linear(in_channels, 20*2)
        self.classifier_lb = nn.Linear(in_channels, 5*2)
        self.classifier_rb = nn.Linear(in_channels, 5*2)


    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x_dict):
        x = x_dict['out4']

        contour_out = self.conv_contour(x) 
        left_eyebrow_out = self.conv_left_eyebrow(x) 
        right_eyebrow_out = self.conv_right_eyebrow(x)
        nose_out = self.conv_nose(x) 
        left_eye_out = self.conv_left_eye(x) 
        right_eye_out = self.conv_right_eye(x) 
        lip_out = self.conv_lip(x) 
        lb = self.conv_left_eyebrow_b(x)
        rb = self.conv_right_eyebrow_b(x)

        contour_out = contour_out.view(contour_out.size(0), -1)
        left_eyebrow_out = left_eyebrow_out.view(left_eyebrow_out.size(0), -1)
        right_eyebrow_out = right_eyebrow_out.view(right_eyebrow_out.size(0), -1)
        nose_out = nose_out.view(nose_out.size(0), -1)
        left_eye_out = left_eye_out.view(left_eye_out.size(0), -1)
        right_eye_out = right_eye_out.view(right_eye_out.size(0), -1)
        lip_out = lip_out.view(lip_out.size(0), -1)
        lb = lb.view(lb.size(0), -1)
        rb = rb.view(rb.size(0), -1)


        contour_out = self.classifier_contour(contour_out) 
        left_eyebrow_out = self.classifier_left_eyebrow(left_eyebrow_out) 
        right_eyebrow_out = self.classifier_right_eyebrow(right_eyebrow_out)
        nose_out = self.classifier_nose(nose_out) 
        left_eye_out = self.classifier_left_eye(left_eye_out) 
        right_eye_out = self.classifier_right_eye(right_eye_out) 
        lip_out = self.classifier_lip(lip_out) 
        lb = self.classifier_lb(lb) 
        rb = self.classifier_rb(rb) 


        out = torch.cat([contour_out, left_eyebrow_out, right_eyebrow_out, nose_out, left_eye_out, right_eye_out, lip_out, lb, rb], 1)

        return out

class MapToNodePredictor(nn.Module):
    def __init__(self, in_channels, feat_size, num_points, **kwargs):
        super(MapToNodePredictor, self).__init__()
        self.map_to_node = MapToNode(in_channels=in_channels, num_points=num_points)
        self.W = nn.Parameter(torch.FloatTensor(feat_size**2, 2)) # 64 x 2

    def forward(self, x_dict):
        out = self.map_to_node(x_dict)
        # W: 64 x 2 out: 98 x 64
        out = torch.matmul(out, self.W) # out: 98 x 2
        out = out.view(out.size(0), -1)

        return out

class ChMapToNodePredictor(nn.Module):
    """docstring for ChMapToNodePredictor"""
    def __init__(self, in_channels, feat_size, num_points, **kwargs):
        super(ChMapToNodePredictor, self).__init__()
        self.map_to_node = MapToNode(in_channels=in_channels, num_points=num_points)
        self.W = nn.Parameter(torch.FloatTensor(2, num_points, feat_size**2))

    def forward(self, x_dict):
        out = self.map_to_node(x_dict)
        # out: N x 1 x 98 x 64
        out = torch.unsqueeze(out, dim=1)
        out = out*self.W
        # out: N x 2 x 98 x 64
        out = torch.sum(out, dim=3)
        out = out.view(out.size(0), -1)
        return out
