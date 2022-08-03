import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F


from timm.models.vision_transformer import Block
from timm.models.vision_transformer_hybrid import _resnetv2
from timm.models.layers import trunc_normal_

from .blocks import *



class SA_DCR_Blocks(nn.Module):
    
    def __init__(self, grid_height=22, grid_width=76, 
                 feature_dim=1024, embed_dim=768, depth=12, num_sa_layer=6,
                 num_heads=12, mlp_ratio=4., qkv_bias=True,  
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 dcr_module=DepthCueRectification_Sp,
                 norm_layer=None, act_layer=None, 
                 device=None,
                ):
        """

        """
        super(SA_DCR_Blocks,self).__init__()
        
        self.gs_h = grid_height or 384//16
        self.gs_w = grid_width or 384//16

        num_patches = self.gs_h * self.gs_w

        patch_size = 1

        self.depth = depth

        self.num_sa = num_sa_layer
        
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        
        self.proj = nn.Conv2d(feature_dim, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.Blocks = nn.Sequential(*[Block(dim=embed_dim, num_heads=num_heads, 
                                            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
                                            drop=drop_rate,attn_drop=attn_drop_rate, drop_path=dpr[i], 
                                            norm_layer=norm_layer, act_layer=act_layer) 
                                      for i in range(depth)])
        
        self.device = device or torch.device('cpu')

        self.register_buffer("coords", get_rel_vectors(self.gs_w, self.gs_h))

        self.DCR = nn.ModuleList(
                                [
                                dcr_module(embed_dim, 
                                int(mlp_ratio*embed_dim), 
                                num_patches, 
                                cls_token=1, 
                                dropout=drop_rate, 
                                device=self.device) for i in range(num_sa_layer, depth)
                                ]
                                )
        
        self.norm = norm_layer(embed_dim)
        
        self.reset_parameters()
        
        
    def reset_parameters(self):
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        
    
    def forward(self, src, tgt, layer_out=[8,11], before_sa=True, pos_supp=True):

        return self.forward_dcr_blocks_flex(src, tgt, layer_out, before_sa, pos_supp)
    
    
    def _resize_pos_embed(self, posemb, gs_h, gs_w):

        posemb_tok, posemb_grid = (
            posemb[:, : self.start_index],
            posemb[0, self.start_index :],
        )

        posemb_grid = posemb_grid.reshape(1, self.gs_h, self.gs_w, -1).permute(0, 3, 1, 2)
        posemb_grid = F.interpolate(posemb_grid, size=(gs_h, gs_w), mode="bilinear")
        posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_h * gs_w, -1)

        posemb = torch.cat([posemb_tok, posemb_grid], dim=1)

        return posemb
    
    
    def forward_dcr_blocks_flex(self, src, tgt, layer_out=[8,11], before_sa=True, pos_supp=False):

        B, _, h, w = src.shape

        if (h, w)!= (self.gs_h, self.gs_w):
            self.coords = get_rel_vectors(w, h).to(self.device)
            pos_emb = self._resize_pos_embed(self.pos_embed, h, w)

        src = self.proj(src).flatten(2).transpose(1, 2)
        tgt = self.proj(tgt).flatten(2).transpose(1, 2)

        cls_token = self.cls_token.expand(B, -1, -1)

        src = torch.cat((cls_token, src), dim = 1) + self.pos_embed
        tgt = torch.cat((cls_token, tgt), dim = 1) + self.pos_embed

        DCR_Blocks = zip(list(range(self.num_sa, self.depth)),
                       self.DCR,
                       self.Blocks[self.num_sa:])

        outputs = []

        for sa_layer in self.Blocks[:self.num_sa]:
            src = sa_layer(src)
            tgt = sa_layer(tgt)

        if before_sa:
            for l_id, dcr_layer, sa_layer in DCR_Blocks:
                src = dcr_layer(src, tgt, self.coords)

                src = sa_layer(src + self.pos_embed) if pos_supp else sa_layer(src)
                tgt = (sa_layer(tgt + self.pos_embed) if pos_supp else sa_layer(tgt)) \
                                                    if l_id < self.depth - 1 else tgt
                
                if l_id in layer_out:
                    outputs.append(src)
        else:
            for l_id, dcr_layer, sa_layer in DCR_Blocks:
                src = sa_layer(src + self.pos_embed) if pos_supp else sa_layer(src)
                tgt = sa_layer(tgt + self.pos_embed) if pos_supp else sa_layer(tgt)

                src = dcr_layer(src, tgt, self.coords)
                
                if l_id in layer_out:
                    outputs.append(src) 

        return outputs
    
    