import copy, math

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer_hybrid import _resnetv2, _create_vision_transformer_hybrid
from .depth_cue_rectification import *



class ChiTransformer(nn.Module):
    def __init__(self, input_size=(3, 384, 384), layer=(3,4,9), embed_dim=768, depth=12, 
                 num_heads=12, mlp_factor=4, pretrained=True, start_layer_id=6,
                 norm_layer=None, act_layer=None, block_fn=None):
        
        super(ChiTransformer, self).__init__()
    
        embedder_stride = 2**(2+len(layer)-1)
        self.patch_size = (embedder_stride,)*2
        self.depth = depth
        self.input_size = input_size
        
        assert input_size[1]%embedder_stride == 0 and input_size[2]%embedder_stride == 0, \
                        f"input width and height should be divisible by {embedder_stride}"
        
        attn_size = (input_size[1]//embedder_stride, input_size[2]//embedder_stride)
        
        backbone = _resnetv2(layer, padding_same=True, stem_type='same')
        
        vit_kwargs = dict(embed_dim=embed_dim, depth=depth, num_heads=num_heads, distilled=False,
                            patch_size=1, num_classes=0, qkv_bias=True, norm_layer=None, act_layer=None)
        
        self.vit = _create_vision_transformer_hybrid('vit_base_r50_s16_384', backbone=backbone, 
                                                        pretrained=pretrained, **vit_kwargs)
        
        self.start_index = self.vit.num_tokens
        
        self.start_layer_id=start_layer_id or depth - depth//2
        dcr_kwargs = dict(dim=embed_dim, d_ff=embed_dim*mlp_factor, 
                          height=attn_size[1], width=attn_size[0], cls_token=1)
        self.DCR_layer = nn.ModuleList([copy.deepcopy(DepthCueRectification_G(**dcr_kwargs)) for i in range(depth-self.start_layer_id)])
        
        
    def forward(self, x, y):
        assert x.shape == y.shape, \
        f'input shapes {x.shape}, {y.shape} are not same.'
        
        B, c, h, w = x.shape
        
        pos_embed = self._resize_pos_embed(h // self.patch_size[1], w // self.patch_size[0])
        
        y = self.vit.patch_embed.backbone(y)
        x = self.vit.patch_embed.backbone(x)
        
        x = self.vit.patch_embed.proj(x).flatten(2).transpose(1, 2)
        y = self.vit.patch_embed.proj(y).flatten(2).transpose(1, 2)
        
        cls_token = self.vit.cls_token.expand(x.shape[0], -1, -1)

        y = torch.cat((cls_token, y), dim=1)
        x = torch.cat((cls_token, x), dim=1)
        
        x = x + pos_embed
        y = y + pos_embed
        
        j=0
        for i in range(self.depth):
            y = self.vit.blocks[i](y)
            x = self.vit.blocks[i](x)
            if i >= self.start_layer_id - 1 and i < self.depth - 1:
                x = self.DCR_layer[j](x, y)
                j += 1
        
        x = self.vit.norm(x)
        
        return x
    
    def _resize_pos_embed(self, new_h, new_w):
        posemb_cls, posemb_grid = (
            self.vit.pos_embed[:, : self.start_index],
            self.vit.pos_embed[0, self.start_index :],
        )
        # old grid size 384x384
        old_size = int(math.sqrt(len(posemb_grid)))

        posemb_grid = posemb_grid.reshape(1, old_size, old_size, -1).permute(0, 3, 1, 2)
        posemb_grid = F.interpolate(posemb_grid, size=(new_h, new_w), mode="bilinear")
        posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, new_h * new_w, -1)

        posemb = torch.cat([posemb_cls, posemb_grid], dim=1)

        return posemb
