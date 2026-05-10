import torch
import torch.nn as nn
from .unet_transformer import UNetTransformer

class Generator(nn.Module):
    '''
    Mask-conditioned generator: input = [image, mask, noise_channels] -> fake image
    For pure mask-conditioning, pass a zero image.
    '''
    def __init__(self, in_image_channels=3, noise_channels=2, base_ch=64, transformer_blocks=1, transformer_heads=4):
        super().__init__()
        self.in_channels = in_image_channels + 1 + noise_channels
        self.unet = UNetTransformer(
            in_channels=self.in_channels,
            num_classes=in_image_channels,
            base_ch=base_ch,
            transformer_blocks=transformer_blocks,
            transformer_heads=transformer_heads,
        )
        self.tanh = nn.Tanh()

    def forward(self, image, mask, noise):
        x = torch.cat([image, mask, noise], dim=1)
        out = self.unet(x)
        return self.tanh(out)

class PatchDiscriminator(nn.Module):
    '''
    PatchGAN discriminator conditioned on mask: D([image, mask]) -> score map
    '''
    def __init__(self, in_image_channels=3, base_ch=64):
        super().__init__()
        in_ch = in_image_channels + 1
        layers = []
        nf = base_ch
        def block(cin, cout, norm=True):
            layers = [nn.Conv2d(cin, cout, 4, stride=2, padding=1)]
            if norm:
                layers.append(nn.InstanceNorm2d(cout, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        layers += block(in_ch, nf, norm=False)
        layers += block(nf, nf*2)
        layers += block(nf*2, nf*4)
        layers += block(nf*4, nf*8)
        layers += [nn.Conv2d(nf*8, 1, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, image, mask):
        x = torch.cat([image, mask], dim=1)
        return self.net(x)  # (B, 1, H', W')
