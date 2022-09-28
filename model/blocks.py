from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from timm.models.layers import trunc_normal_

def get_abscoords(width, height):
    
    return torch.stack(torch.meshgrid(torch.arange(width), 
                                      torch.arange(height), 
                                      indexing='xy'), dim=0).type(torch.float)


def get_rel_vectors(width, height):
    
    abscoords = get_abscoords(width, height)
    
    rel_coords = abscoords.flatten(1).permute(1, 0).contiguous()
    
    #rel_coords: [N, 1, 2] - [1, N, 2]
    rel_coords = rel_coords.unsqueeze(-3) - rel_coords.unsqueeze(-2)  #[N, N, 2=(row/y, col/x)]
    rel_vectors = torch.cat(
                        (torch.ones(tuple(rel_coords.shape[:2])+(1,)),
                         rel_coords, 
                         rel_coords.prod(dim=-1, keepdim=True),
                         rel_coords[:, :, 0, None] ** 2, 
                         rel_coords[:, :, 1, None] ** 2), 
                            -1)

    return rel_vectors


class CrossAttention_G(nn.Module): 
    """
    """ 
    
    def __init__(self, dim, num_patches, a_temp = False):
        
        super(CrossAttention_G, self).__init__()

        self.qk = nn.Linear(dim, dim, bias=False)
        
        self.gating = nn.Parameter(torch.tensor(1.))
        self.h_temp = nn.Parameter(torch.tensor(0.1))
        self.p_temp = nn.Parameter(torch.tensor(1.))
        self.pos_emb = nn.Parameter(torch.zeros(num_patches, 6, 1))
        self.scale = nn.Parameter(torch.tensor(2 * dim ** -0.5)) if a_temp else dim ** -0.5

        self.reset_parameters()
        
    def reset_parameters(self):
        trunc_normal_(self.pos_emb, a=0., b=1.)

    def forward(self, x, y, coords):
        
        #src:[B, N, d]， tgt:[B, N, d]
        attn = self.compute_attention(x, y, coords)  # [B, N, N]
        heat_map = self.get_heat_map(attn)  # [B, N, 1]
        
        return attn @ y, heat_map

    def compute_attention(self, x, y, coords):
        #src:[B, N, d]， tgt:[B, N, d]
        B = x.shape[0]
        
        k = self.qk(y)
        
        patch_score = (x @ k.transpose(-2, -1)) * self.scale  # [B, N, N]
        patch_score = patch_score.softmax(dim=-1)
        
        pos_score = (-self.p_temp * coords.matmul(self.pos_emb)).squeeze(-1).softmax(dim=-1).expand(B, -1, -1) 
        
        attn = (1.-torch.sigmoid(self.gating)) * patch_score + \
                                    torch.sigmoid(self.gating) * pos_score  # [B, N, N]

        return attn

    def get_heat_map(self, attn):

        entropy = torch.sum(-attn * torch.log(attn + 1e-8), dim=-1, keepdim=True) 
        hmap = 2*(1 - torch.sigmoid(self.h_temp * entropy))

        return hmap # [B, N, 1]


class SpectralDecomp(nn.Module):
    """
    """
    
    def __init__(self, dim):
        
        super(SpectralDecomp, self).__init__()
        
        self.U = nn.Parameter(torch.empty(dim, dim))
        
        self.S1 = nn.Parameter(torch.empty(dim, 1))
        self.S2 = nn.Parameter(torch.empty(dim, 1))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        init.kaiming_uniform_(self.S1, nonlinearity='relu')
        init.kaiming_uniform_(self.S2, nonlinearity='relu')
        init.xavier_uniform_(self.U)
        
    def forward(self, x):
        #x: [B, N, d]
        x1 = x @ self.U.T @ self.S1.squeeze(-1).abs().diag() @ self.U #[B, N, d]
        x2 = x @ self.U.T @ self.S2.squeeze(-1).abs().diag() @ self.U
        
        return x1, x2

    
class CrossAttention_Sp(nn.Module):
    """
    """
    
    def __init__(self, dim, num_patches, device=None, a_temp=False):
        
        super(CrossAttention_Sp, self).__init__()
        
        self.spectrum = SpectralDecomp(dim)
        
        self.gating = nn.Parameter(torch.tensor(1.))
        self.h_temp = nn.Parameter(torch.tensor(0.1))
        self.p_temp = nn.Parameter(torch.tensor(1.))
        self.pos_emb = nn.Parameter(torch.zeros(num_patches, 6, 1))
        self.scale = nn.Parameter(torch.tensor(2 * dim ** -0.5)) if a_temp else dim ** -0.5
        
        self.device = device or torch.device('cpu')
        self.reset_parameters()
        
    def reset_parameters(self):
        trunc_normal_(self.pos_emb, a=0., b=1.)

    def forward(self, x, y, coords):
        
        #src:[B, N, d]， tgt:[B, N, d]
        attn = self.compute_attention(x, y, coords)  # [B, 2, N, N]
        
        heat_map, route = self.get_heat_map(attn)  # [B, N, 2]
        heat_map = (heat_map * route.detach()).sum(dim=-1, keepdim=True) #[B, N, 1]
        
        attn = (attn * route.transpose(-1, -2)[...,None]).sum(dim=1) # [B, N, N]
        x = attn @ y  # [B, N, d]
    
        return x, heat_map #[B, N, d], [B, N, 1]

    def compute_attention(self, x, y, coords):
        
        #src:[B, N, d]， tgt:[B, N, d]
        B = x.shape[0]
        x = torch.stack(self.spectrum(x), dim=1)  # [B, 2, N, d]
        
        patch_score = ((x @ y.unsqueeze(1).transpose(-2, -1)) * \
                       (self.scale)).softmax(dim=-1)  # [B, 2, N, N]

        pos_score = (-self.p_temp.abs() * coords.matmul(self.pos_emb)).squeeze(-1).softmax(dim=-1).expand(B, 2, -1, -1)
        
        attn = (1.-torch.sigmoid(self.gating)) * patch_score + \
                                    torch.sigmoid(self.gating) * pos_score  # [B, 2, N, N]

        return attn

    def get_heat_map(self, attn):
        
        entropy = torch.sum(-attn * torch.log(attn + 1e-8), dim=-1).permute(0,2,1)

        hmap = 2 - 2*torch.sigmoid(self.h_temp * entropy) #[B, N, 2]

        fgbg = hmap[..., 0] >= hmap[..., 1]
        
        route = torch.zeros_like(hmap)
        route[...,0][fgbg] += 1.
        route[...,1][~fgbg] += 1.
        
        return hmap, route.requires_grad_(False)  #[B, N, 2]


class DepthCueRectification_G(nn.Module):
    """
    """
    
    def __init__(self, dim, d_ff, num_patches, cls_token=1, dropout=0., 
                 device=None, layer_norm=False):
        
        super().__init__()
        
        self.cls = cls_token
        
        self.crs_layer = CrossAttention_G(dim, num_patches)

        if layer_norm:
            self.layer_norm = nn.LayerNorm(dim)
        else:
            self.layer_norm=nn.Identity()
        
        self.proj = nn.Sequential(nn.Linear(2*dim, d_ff),
                                  nn.GELU(),
                                  nn.Dropout(dropout),
                                  nn.Linear(d_ff, dim),
                                  nn.Dropout(dropout),
                                 )

        self.dropout = nn.Dropout(dropout)

        self.device = device or torch.device('cpu')
        
    def forward(self, x, y, coords):
        
        x = self.layer_norm(x)
        y = self.layer_norm(y)
        
        y_cls = y[:, :self.cls, :]
        
        y, heat_map = self.crs_layer(x[:,self.cls:,:], y[:,self.cls:,:], coords)
        y = torch.cat((y_cls, y), 1)
        
        x_p = self.proj(torch.concat((x, y), dim=-1))
        
        x_p = torch.cat((x_p[:,:self.cls,:], heat_map * x_p[:,self.cls:,:]), 1)
        
        return x + self.dropout(x_p)  

    
class DepthCueRectification_Sp(DepthCueRectification_G):
    """
    """
    
    def __init__(self, dim, d_ff, num_patches, 
                **kwargs):
        
        super().__init__(dim, d_ff, num_patches, 
                         **kwargs)
        
        self.crs_layer = CrossAttention_Sp(dim, num_patches, kwargs["device"])
