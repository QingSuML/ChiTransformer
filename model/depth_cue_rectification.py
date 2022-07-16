import torch
import torch.nn as nn
import torch.nn.init as init
from .environ import DEVICE

device = torch.device(DEVICE)

class CrossAttention_G(nn.Module):
    def __init__(self, dim, height, width):
        super(CrossAttention_G, self).__init__()
        self.qk = nn.Linear(dim, dim)
        self.focus = nn.Parameter(torch.tensor(1.))
        self.gating = nn.Parameter(torch.tensor(1.))
        self.temp = nn.Parameter(torch.tensor(0.1))
        abs_coords_y = torch.arange(height).expand(width, -1).T.flatten().type(torch.float)
        rel_coords_y = (- abs_coords_y.unsqueeze(-1) + abs_coords_y).contiguous()
        self.register_buffer('rel_coords_y', rel_coords_y)

    def reset_parameter(self):
        nn.init.constant_(self.focus, 1.)
        nn.init.constant_(self.gating, 1.)
        nn.init.constant_(self.temp, .1)
        nn.init.normal_(self.qk.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.qk.bias)

    def forward(self, x, y):
        #src:[B, N, d]， tgt:[B, N, d]
        B, N, d = x.shape
        attn = self.compute_attention(x, y)  # [B, N, N]
        heat_map = self.get_heat_map(attn)  # [B, N, 1]
        return attn @ y, heat_map

    def compute_attention(self, x, y):
        #src:[B, N, d]， tgt:[B, N, d]
        B, N, d = x.shape
        k = self.qk(y)
        patch_score = x @ k.transpose(-2, -1) * d ** -0.5  # [B, N, N]
        patch_score = patch_score.softmax(dim=-1)
        pos_score = (- self.focus.abs()*torch.square(self.rel_coords_y)).softmax(dim=-1).expand(B, -1, -1) 
        attn = (1.-torch.sigmoid(self.gating)) * patch_score + torch.sigmoid(self.gating) * pos_score  # [B, N, N]
        attn /= attn.sum(dim=-1, keepdims=True)
        return attn

    def get_heat_map(self, attn):
        entropy = torch.sum(-attn * torch.log(attn + 1e-8), dim=-1, keepdim=True) 
        hmap = 2 - 2*torch.sigmoid(self.temp * entropy)
        return hmap # [B, N, 1]

    
class SpectralDecomp(nn.Module):
    def __init__(self, dim):
        super(SpectralDecomp, self).__init__()
        self.U = nn.Parameter(torch.empty(dim,dim))
        self.S1 = nn.Parameter(torch.empty(dim))
        self.S2 = nn.Parameter(torch.empty(dim))
        self.reset_parameters()
        
    def reset_parameters(self):
        init.constant_(self.S1, 1.)
        init.constant_(self.S2, 1.)
        init.orthogonal_(self.U, gain=init.calculate_gain('linear'))
        
    def forward(self, x):
        #x: [B, N, d]
        x1 = x @ self.U.T @ self.S1.square().diag() @ self.U #[B, N, d] # or abs
        x2 = x @ self.U.T @ self.S2.square().diag() @ self.U
        return [x1, x2]

    
class CrossAttention_Sp(nn.Module):
    def __init__(self, dim, height, width):
        super(CrossAttention_Sp, self).__init__()
        self.spectrum = SpectralDecomp(dim)
        self.focus = nn.Parameter(torch.tensor(1.))
        self.gating = nn.Parameter(torch.tensor(1.))
        self.temp = nn.Parameter(torch.tensor(0.1))
        abs_coords_y = torch.arange(height).expand(width, -1).T.flatten().type(torch.float)
        rel_coords_y = (- abs_coords_y.unsqueeze(-1) + abs_coords_y).contiguous()
        self.register_buffer('rel_coords_y', rel_coords_y)

    def forward(self, x, y):
        #src:[B, N, d]， tgt:[B, N, d]
        attn = self.compute_attention(x, y)  # [B, 2, N, N]
        heat_map, route = self.get_heat_map(attn)  # [B, N, 2]
        heat_map = (heat_map * route).sum(dim=-1, keepdim=True) #[B, N, 1]
        x = (attn @ y.unsqueeze(1)).transpose(1, 2)  # [B, N, 2, d]
        x = (x * route[...,None]).sum(dim = -2) #[B, N, d]
        return x, heat_map

    def compute_attention(self, x, y):
        #src:[B, N, d]， tgt:[B, N, d]
        B, N, d = x.shape
        x = torch.stack(self.spectrum(x), dim=1)  # [B, 2, N, d]
        patch_score = ((x @ y.unsqueeze(1).transpose(-2, -1)) * (d ** -0.5)).softmax(dim=-1)  # [B, 2, N, N]
        pos_score = (-self.focus.abs() * torch.square(self.rel_coords_y)).softmax(dim=-1).expand(B, 2, -1, -1)
        attn = (1.-torch.sigmoid(self.gating)) * patch_score + torch.sigmoid(self.gating) * pos_score  # [B, 2, N, N]
        attn /= attn.sum(dim=-1, keepdims=True)
        return attn

    def get_heat_map(self, attn):
        entropy = torch.sum(-attn * torch.log(attn + 1e-8), dim=-1).permute(0,2,1)
        hmap = 2 - 2*torch.sigmoid(self.temp * entropy) # [B, N, 2]
        route = torch.ones(hmap.shape).to(device)
        fgobg = hmap[...,0] >= hmap[..., 1]
        route[fgobg] = torch.tensor([1., 0.]).to(device)
        route[~fgobg] = torch.tensor([0., 1.]).to(device)
        return hmap, route  #[B, N, 2]
    

class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super(LayerNorm, self).__ini__()
        self.a = nn.Parameter(torch.ones(dim))
        self.b = nn.Parameter(torch.zeros(dim))
        self.eps = eps
        
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a * (x - mean)/(std + self.eps) + self.b
    
    
class DepthCueRectification_G(nn.Module):
    """
        CA Rectification layer
    """
    
    def __init__(self, dim, d_ff, height, width, cls_token=1, dropout=0.):
        super(DepthCueRectification_G, self).__init__()
        self.crs_layer = CrossAttention_G(dim, height, width)
        #self.layer_norm = LayerNorm(dim)
        self.w_1 = nn.Linear(2*dim, d_ff)
        self.w_2 = nn.Linear(d_ff, dim)
        self.cls = cls_token
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, y):
        #x = self.layer_norm(x)
        #y = self.layer_norm(y)
        y_t, heat_map = self.crs_layer(x[:,self.cls:,:], y[:,self.cls:,:])
        y = torch.cat((y[:, :self.cls, :], y_t), 1)
        x_p = self.w_1(torch.concat([x, y], dim=-1))
        x_p = self.w_2(self.dropout(x_p.relu()))
        x_p = torch.cat((x_p[:,:self.cls,:], heat_map * x_p[:,self.cls:,:]), 1)
        return x + self.dropout(x_p)  
    

class DepthCueRectification_Sp(DepthCueRectification_G):
    def __init__(self, dim, d_ff, height, width, cls_token=1, dropout=0.):
        super(DepthCueRectification_Sp, self).__init__(dim, d_ff, height, width, cls_token=cls_token, dropout=dropout)
        self.crs_layer = CrossAttention_Sp(dim, height, width)