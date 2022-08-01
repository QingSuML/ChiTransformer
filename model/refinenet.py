import copy
from doctest import OutputChecker
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F



class DepthRefineNet(nn.Module):

    def __init__(self, non_negative=True, invert=True,
                 scale=1.0, shift=0.0,
                 in_shape=[256, 512, 768, 768], out_shape=256, 
                 vit_features=768, size=[352, 1216], features=256, 
                 use_bn=False):
        """

        """
        
        super().__init__()
        
        self.invert = invert
        self.scale = scale
        self.shift = shift
        start_index=1
        
        #act_postprocess3,4
        readout_oper= [ProjectReadout(vit_features, start_index) for _ in in_shape[2:]]
        
        self.sa_postprocess_2 = nn.Sequential(readout_oper[0], 
                                              Transpose(1, 2),
                                              nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
                                              nn.Conv2d(in_channels=vit_features, out_channels=in_shape[2],
                                                        kernel_size=1, stride=1, padding=0,),
                                             )
        
        self.sa_postprocess_3 = nn.Sequential(readout_oper[1],
                                              Transpose(1, 2),
                                              nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
                                              nn.Conv2d(in_channels=vit_features, out_channels=in_shape[3], 
                                                        kernel_size=1, stride=1, padding=0,),
                                              nn.Conv2d(in_channels=in_shape[3], out_channels=in_shape[3], 
                                                        kernel_size=3, stride=2, padding=1,),
                                             )
        
        #layer1_rn, layer2_rn, ..., layer4_rn 
        self.proj_rn = nn.ModuleList([nn.Conv2d(in_shape[i], out_shape,
                                                kernel_size=3, stride=1, padding=1,
                                                bias=False, groups=1,)
                                      for i in range(len(in_shape))])
        
        #scratch.refinenet1,2,3,4
        self.refinenet = nn.ModuleList([FeatureFusionBlock(features,nn.ReLU(False),
                                                           bn=use_bn,
                                                           align_corners=True,)
                                        for i in range(len(in_shape))])
        
        self.head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
            nn.Identity(),)
        

    def forward(self,inputs):

        output = {}
        
        input0 = inputs[0]
        input1 = inputs[1]

        input2 = self.sa_postprocess_2(inputs[2])
        input3 = self.sa_postprocess_3(inputs[3])
        
        rp_0 = self.proj_rn[0](input0)
        rp_1 = self.proj_rn[1](input1)
        rp_2 = self.proj_rn[2](input2)
        rp_3 = self.proj_rn[3](input3)
            
        rf_3 = self.refinenet[3](rp_3)
        rf_2 = self.refinenet[2](rf_3, rp_2)
        rf_1 = self.refinenet[1](rf_2, rp_1)
        rf_0 = self.refinenet[0](rf_1, rp_0)
        
        depth = self.head(rf_0)
        
        if self.invert:
            depth = self.depth_calib(depth, self.scale, self.shift)
        
        output[("depth", 0)] = depth.squeeze(1)
        
        return output
    

    def depth_calib(self, inv_depth, scale, shift):
        
        depth = scale * inv_depth + shift
        depth[depth < 1e-8] = 1e-8
        depth = 1.0 / depth
        
        return depth
        
        

class ProjectReadout(nn.Module):
    
    def __init__(self, in_features, start_index=1):

        """
        """
        
        super(ProjectReadout, self).__init__()
        
        self.start_index = start_index

        self.project = nn.Sequential(nn.Linear(2 * in_features, in_features), nn.GELU())

    def forward(self, x):
        
        readout = x[:, 0].unsqueeze(1).expand_as(x[:, self.start_index :])
        features = torch.cat((x[:, self.start_index :], readout), -1)

        return self.project(features)


class Interpolate(nn.Module):

    def __init__(self, scale_factor, mode, align_corners=False):

        super(Interpolate, self).__init__()

        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):

        x = F.interpolate(x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,)

        return x


class ResidualConvUnit(nn.Module):

    def __init__(self, features, activation, bn):

        super(ResidualConvUnit, self).__init__()

        self.bn = bn

        self.conv1 = nn.Conv2d(features, features, 
                               kernel_size=3, stride=1, padding=1,
                               bias=not self.bn, groups=1,)
        self.conv2 = copy.deepcopy(self.conv1)

        if self.bn:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):

        out = self.activation(x)
        out = self.conv1(out)
        if self.bn:
            out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):

    def __init__(self, features, activation,
                bn=False, align_corners=True,):
        """
        
        """
        super(FeatureFusionBlock, self).__init__()

        self.align_corners = align_corners
        
        out_features = features

        self.out_conv = nn.Conv2d(features, out_features, 
                                  kernel_size=1, stride=1, padding=0, 
                                  bias=True, groups=1,)

        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs):
        
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        output = F.interpolate(output, 
                               scale_factor=2, mode="bilinear", 
                               align_corners=self.align_corners)

        output = self.out_conv(output)

        return output

        
class Transpose(nn.Module):
    
    def __init__(self, dim0, dim1):
        
        super(Transpose, self).__init__()
        
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        
        x = x.transpose(self.dim0, self.dim1)
        
        return x