import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    '''Two 3x3 convs + BN + ReLU'''
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.act   = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        return x

class UpBlock(nn.Module):
    '''Upsample + ConvBlock'''
    def __init__(self, in_ch, out_ch, mode="bilinear"):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode=mode, align_corners=(mode=="bilinear"))
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class PositionalEncoding2D(nn.Module):
    '''Simple 2D sine-cos positional encoding (added to channels).'''
    def __init__(self, channels):
        super().__init__()
        if channels % 4 != 0:
            raise ValueError("Channels must be divisible by 4 for 2D positional encoding.")
        self.channels = channels

    def forward(self, x):
        b,c,h,w = x.shape
        device = x.device
        y_pos = torch.linspace(-1, 1, steps=h, device=device).unsqueeze(1).repeat(1, w)
        x_pos = torch.linspace(-1, 1, steps=w, device=device).unsqueeze(0).repeat(h, 1)
        pe = []
        div_term = torch.arange(0, c//4, device=device).float()
        div_term = torch.pow(10000, (2*div_term)/ (c//2))
        for fn in (torch.sin, torch.cos):
            pe_y = fn(y_pos.unsqueeze(0).unsqueeze(0) / div_term.view(1,-1,1,1))
            pe_x = fn(x_pos.unsqueeze(0).unsqueeze(0) / div_term.view(1,-1,1,1))
            pe.append(pe_y)
            pe.append(pe_x)
        pe = torch.cat(pe, dim=1)[:,:c,:,:]
        return x + pe

class TransformerBlock2D(nn.Module):
    '''
    Lightweight transformer encoder over flattened HW tokens with MultiheadAttention.
    Input/Output: (B, C, H, W). C is the embed dim.
    '''
    def __init__(self, embed_dim, num_heads=4, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.pos = PositionalEncoding2D(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.pos(x)
        x_flat = x.flatten(2).permute(2,0,1)  # (L=HW, B, C)
        x_attn = self.attn(self.norm1(x_flat), self.norm1(x_flat), self.norm1(x_flat), need_weights=False)[0]
        x = x_flat + x_attn
        x = x + self.mlp(self.norm2(x))
        x = x.permute(1,2,0).view(b, c, h, w)
        return x