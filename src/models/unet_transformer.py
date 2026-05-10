import torch
import torch.nn as nn
from .blocks import ConvBlock, UpBlock, TransformerBlock2D

class UNetTransformer(nn.Module):
    '''
    U-Net with a Transformer block in the bottleneck (and optional extra blocks).
    '''
    def __init__(self, in_channels=3, num_classes=1, base_ch=64, transformer_blocks=1, transformer_heads=4):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch*2)
        self.enc3 = ConvBlock(base_ch*2, base_ch*4)
        self.enc4 = ConvBlock(base_ch*4, base_ch*8)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base_ch*8, base_ch*16)

        self.tr_blocks = nn.Sequential(*[TransformerBlock2D(base_ch*16, num_heads=transformer_heads) for _ in range(transformer_blocks)])

        self.up1 = UpBlock(base_ch*16 + base_ch*8, base_ch*8)
        self.up2 = UpBlock(base_ch*8 + base_ch*4, base_ch*4)
        self.up3 = UpBlock(base_ch*4 + base_ch*2, base_ch*2)
        self.up4 = UpBlock(base_ch*2 + base_ch, base_ch)

        self.out = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b  = self.bottleneck(self.pool(e4))
        b  = self.tr_blocks(b)

        d1 = self.up1(b, e4)
        d2 = self.up2(d1, e3)
        d3 = self.up3(d2, e2)
        d4 = self.up4(d3, e1)

        return self.out(d4)
