from functools import partial
import types
import copy

import torch
import torch.nn as nn

from timm.models.layers import StdConv2dSame
from timm.models.resnetv2 import ResNetV2

from .blocks import *
from .dcr import SA_DCR_Blocks
from .refinenet import DepthRefineNet


class Embedder(nn.Module):
    
    def __init__(self, in_chans=3, layers=(3,4,9)):
        
        super().__init__()
        
        conv_layer = partial(StdConv2dSame, eps=1e-8)
        stem_type = "same"
        
        self.num_stage=len(layers)
        
        self.backbone = ResNetV2(layers=layers, num_classes=0, 
                                 global_pool='', in_chans=in_chans, 
                                 preact=False, stem_type=stem_type, 
                                 conv_layer=conv_layer)
        
    def forward(self, x, layer_out=[0,1]):
        
        outputs = []
        
        x = self.backbone.stem(x)
        
        for i in range(self.num_stage):
            x = self.backbone.stages[i](x)
            if i in layer_out:
                outputs.append(x)
                
        if not self.num_stage - 1 in layer_out:
            outputs.append(x)
            
        return outputs
    
    def forward_one(self, x):
        
        x = self.backbone.stem(x)
        
        for i in range(self.num_stage):
            x = self.backbone.stages[i](x)
        
        return x
    
    
    
class ChitransformerDepth(nn.Module):
    """
    """
    def __init__(self,
                 in_chans=3, 
                 embed_layer=(3, 4, 9),
                 in_chans_sa=1024,
                 embed_dim=768,
                 depth=12,
                 sa_depth=6,
                 dcr_module=DepthCueRectification_Sp,
                 invert=True,
                 scale=None,
                 shift=None,
                 size=[352, 1216], 
                 device=None,
                ):
        
        super().__init__()
        
        self.patch_embedder = Embedder(in_chans=in_chans, layers=embed_layer)
        
        device = device or torch.device('cpu')
        
        self.sa_dcr = SA_DCR_Blocks(feature_dim=in_chans_sa, 
                                    embed_dim=embed_dim, 
                                    depth=depth, 
                                    num_sa_layer=sa_depth,
                                    dcr_module=dcr_module,
                                    device=device,
                                   )
        
        scale = scale or 1.0
        shift = shift or 0.0
        
        self.refinenet = DepthRefineNet(invert=invert,
                                        scale=scale, 
                                        shift=shift,
                                        in_shape=[256, 512, embed_dim, embed_dim], 
                                        out_shape=256, 
                                        vit_features=embed_dim, 
                                        size=[352, 1216],
                                       )
        
        self.layer_out = [0, 1, 8, 11]
        
    def forward(self, iml, imr):

        iml = self.patch_embedder(iml, layer_out=self.layer_out[:2])
        imr = self.patch_embedder.forward_one(imr)
        cuel = self.sa_dcr(iml[-1], imr, layer_out=self.layer_out[2:])
        depth = self.refinenet([iml[0], iml[1], cuel[0], cuel[1]]) 
        
        return depth
        

        
class ChitransformerDepth_MS(ChitransformerDepth):
    """
    """
    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)
        
        output_scale = [0,1,2,3]
        self.refinenet.head = nn.ModuleList([copy.deepcopy(self.refinenet.head) for _ in output_scale])
        self.refinenet.forward = types.MethodType(refinenet_ms_forward, self.refinenet)
        
    def forward(self, iml, imr):
        iml = self.patch_embedder(iml, layer_out=self.layer_out[:2])
        imr = self.patch_embedder.forward_one(imr)
        
        cuel = self.sa_dcr(iml[-1], imr, layer_out=self.layer_out[2:])
        depth = self.refinenet([iml[0], iml[1], cuel[0], cuel[1]]) 
        
        return depth
        
        
def refinenet_ms_forward(self, inputs):
    
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
    
    depth = {}
    depth[("depth", 0)] = self.head[0](rf_0)
    depth[("depth", 1)] = self.head[1](rf_1)
    depth[("depth", 2)] = self.head[2](rf_2)
    depth[("depth", 3)] = self.head[3](rf_3)

    if self.invert:
        depth[("depth", 0)] = self.depth_calib(depth[("depth", 0)], self.scale, self.shift)
        depth[("depth", 1)] = self.depth_calib(depth[("depth", 1)], self.scale, self.shift)
        depth[("depth", 2)] = self.depth_calib(depth[("depth", 2)], self.scale, self.shift)
        depth[("depth", 3)] = self.depth_calib(depth[("depth", 3)], self.scale, self.shift)
    
    for values in depth.values():
        values.squeeze(1)

    return depth
    
        