import torch
from torch import nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """双重卷积模块，与您的U-Net中相同"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """下采样模块，与您的U-Net中相同"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """上采样模块，用于UNet++的密集连接"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # 处理尺寸不匹配问题
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class NestedDoubleConv(nn.Module):
    """用于UNet++节点的双重卷积模块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        
    def forward(self, x):
        return self.conv(x)

class UNetPlusPlus(nn.Module):
    """UNet++实现
    
    特点：
    1. 密集跳跃连接：每个解码器节点接收来自多个编码器节点的特征
    2. 深度监督：多个输出层，可用于训练监督
    3. 更密集的特征复用
    """
    def __init__(self, n_channels=14, n_classes=1, deep_supervision=True, bilinear=True):
        super(UNetPlusPlus, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.deep_supervision = deep_supervision
        self.bilinear = bilinear
        
        # 初始卷积
        self.conv0_0 = DoubleConv(n_channels, 64)
        
        # 编码器路径
        self.conv1_0 = Down(64, 128)
        self.conv2_0 = Down(128, 256)
        self.conv3_0 = Down(256, 512)
        self.conv4_0 = Down(512, 1024)
        
        # 中间层（编码器-解码器连接）
        # 第1列
        self.conv0_1 = NestedDoubleConv(64 + 128, 64)
        # 第2列
        self.conv1_1 = NestedDoubleConv(128 + 256, 128)
        self.conv0_2 = NestedDoubleConv(64 + 128 + 128, 64)
        # 第3列
        self.conv2_1 = NestedDoubleConv(256 + 512, 256)
        self.conv1_2 = NestedDoubleConv(128 + 256 + 256, 128)
        self.conv0_3 = NestedDoubleConv(64 + 128 + 128 + 128, 64)
        # 第4列
        self.conv3_1 = NestedDoubleConv(512 + 1024, 512)
        self.conv2_2 = NestedDoubleConv(256 + 512 + 512, 256)
        self.conv1_3 = NestedDoubleConv(128 + 256 + 256 + 256, 128)
        self.conv0_4 = NestedDoubleConv(64 + 128 + 128 + 128 + 128, 64)
        
        # 上采样模块（用于密集连接）
        self.up1 = Up(128, 64, bilinear)
        self.up2 = Up(256, 128, bilinear)
        self.up3 = Up(512, 256, bilinear)
        self.up4 = Up(1024, 512, bilinear)
        
        # 深度监督输出层（如果启用）
        if deep_supervision:
            self.final1 = nn.Conv2d(64, n_classes, kernel_size=1)
            self.final2 = nn.Conv2d(64, n_classes, kernel_size=1)
            self.final3 = nn.Conv2d(64, n_classes, kernel_size=1)
            self.final4 = nn.Conv2d(64, n_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(64, n_classes, kernel_size=1)
            
    def forward(self, x):
        # 编码器路径
        x0_0 = self.conv0_0(x)  # [batch, 64, H, W]
        x1_0 = self.conv1_0(x0_0)  # [batch, 128, H/2, W/2]
        x2_0 = self.conv2_0(x1_0)  # [batch, 256, H/4, W/4]
        x3_0 = self.conv3_0(x2_0)  # [batch, 512, H/8, W/8]
        x4_0 = self.conv4_0(x3_0)  # [batch, 1024, H/16, W/16]
        
        # 密集跳跃连接路径
        # 第1列
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up1(x1_0)], dim=1))
        
        # 第2列
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up2(x2_0)], dim=1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up1(x1_1)], dim=1))
        
        # 第3列
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up3(x3_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up2(x2_1)], dim=1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up1(x1_2)], dim=1))
        
        # 第4列
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up4(x4_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up3(x3_1)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up2(x2_2)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up1(x1_3)], dim=1))
        
        # 输出
        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            
            # 训练时返回所有输出用于深度监督，测试时通常只返回最后一个
            if self.training:
                return output1, output2, output3, output4
            else:
                return output4
        else:
            return self.final(x0_4)


# 简化的UNet++版本（如果觉得上面的太复杂，这里提供一个简化版）
class SimplifiedUNetPlusPlus(nn.Module):
    """简化的UNet++版本，更容易理解和实现"""
    def __init__(self, n_channels=14, n_classes=1, bilinear=True):
        super(SimplifiedUNetPlusPlus, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # 编码器（与U-Net相同）
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # 解码器 - 使用密集连接
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # 额外的密集连接卷积块
        self.conv_dense1 = DoubleConv(64 + 64, 64)  # 连接x0_0和上采样后的x1_1
        self.conv_dense2 = DoubleConv(64 + 64 + 64, 64)  # 连接x0_0, x0_1和上采样后的x1_2
        self.conv_dense3 = DoubleConv(64 + 64 + 64 + 64, 64)  # 连接x0_0, x0_1, x0_2和上采样后的x1_3
        
        # 输出卷积
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
        
    def forward(self, x):
        # 编码器路径
        x1 = self.inc(x)  # [batch, 64, H, W]
        x2 = self.down1(x1)  # [batch, 128, H/2, W/2]
        x3 = self.down2(x2)  # [batch, 256, H/4, W/4]
        x4 = self.down3(x3)  # [batch, 512, H/8, W/8]
        x5 = self.down4(x4)  # [batch, 1024, H/16, W/16]
        
        # 解码器路径，带有密集连接
        # 第一层上采样
        u1 = self.up1(x5, x4)  # [batch, 512, H/8, W/8]
        
        # 第二层上采样
        u2 = self.up2(u1, x3)  # [batch, 256, H/4, W/4]
        
        # 第三层上采样
        u3 = self.up3(u2, x2)  # [batch, 128, H/2, W/2]
        
        # 第四层上采样（标准U-Net路径）
        u4_standard = self.up4(u3, x1)  # [batch, 64, H, W]
        
        # UNet++密集连接路径
        # 第一层密集连接：连接x1和上采样后的u3
        u3_up = F.interpolate(u3, size=x1.shape[2:], mode='bilinear', align_corners=True)
        dense1 = self.conv_dense1(torch.cat([x1, u3_up], dim=1))
        
        # 第二层密集连接：连接x1, dense1和上采样后的u2
        u2_up = F.interpolate(u2, size=x1.shape[2:], mode='bilinear', align_corners=True)
        dense2 = self.conv_dense2(torch.cat([x1, dense1, u2_up], dim=1))
        
        # 第三层密集连接：连接x1, dense1, dense2和上采样后的u1
        u1_up = F.interpolate(u1, size=x1.shape[2:], mode='bilinear', align_corners=True)
        dense3 = self.conv_dense3(torch.cat([x1, dense1, dense2, u1_up], dim=1))
        
        # 最终输出 - 可以选择使用标准路径或密集路径
        # 这里使用最密集的连接作为最终特征
        logits = self.outc(dense3)
        
        return logits


# 测试代码
if __name__ == "__main__":
    # 测试标准UNet++
    model = UNetPlusPlus(n_channels=14, n_classes=1, deep_supervision=True)
    input_tensor = torch.randn(2, 14, 256, 256)  # batch=2, channels=14, H=256, W=256
    output = model(input_tensor)
    print(f"UNet++ 输出形状: {output.shape}")
    
    # 测试简化版UNet++
    model_simple = SimplifiedUNetPlusPlus(n_channels=14, n_classes=1)
    output_simple = model_simple(input_tensor)
    print(f"简化版UNet++ 输出形状: {output_simple.shape}")