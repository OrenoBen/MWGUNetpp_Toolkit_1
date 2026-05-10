import torch
import torch.nn as nn
import torch.nn.functional as F

# ==================== 基础模块 ====================

class DoubleConv(nn.Module):
    """双重卷积模块（与U-Net保持一致）"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
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
    """下采样模块"""
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """上采样模块"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 处理尺寸不匹配
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# ==================== 注意力模块 ====================

class ChannelSpatialAttention(nn.Module):
    """
    通道-空间协同注意力模块
    同时关注重要的通道和空间位置
    """
    def __init__(self, channels, reduction=16):
        super(ChannelSpatialAttention, self).__init__()
        
        # 通道注意力（类似SE模块）
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        
        # 空间注意力
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(channels, 1, 7, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 通道注意力
        channel_weight = self.channel_attn(x)
        x_channel = x * channel_weight
        
        # 空间注意力
        spatial_weight = self.spatial_attn(x_channel)
        x_spatial = x_channel * spatial_weight
        
        return x_spatial


class CrossModalAttention(nn.Module):
    """
    跨模态注意力模块
    让两个模态的特征相互指导、互补
    """
    def __init__(self, channels):
        super(CrossModalAttention, self).__init__()
        
        # 投影到相同的低维特征空间
        self.optical_proj = nn.Conv2d(channels, channels // 4, 1)
        self.terrain_proj = nn.Conv2d(channels, channels // 4, 1)
        
        # 注意力计算
        self.attention_optical = nn.Conv2d(channels // 4, 1, 1)
        self.attention_terrain = nn.Conv2d(channels // 4, 1, 1)
        
        # 特征融合
        self.fusion_optical = nn.Conv2d(channels + channels // 4, channels, 1)
        self.fusion_terrain = nn.Conv2d(channels + channels // 4, channels, 1)
    
    def forward(self, optical_feat, terrain_feat):
        """
        前向传播
        Args:
            optical_feat: 光学特征 [B, C, H, W]
            terrain_feat: 地形特征 [B, C, H, W]
        Returns:
            optical_fused: 融合了地形信息的增强光学特征
            terrain_fused: 融合了光学信息的增强地形特征
        """
        
        # 投影到低维空间
        optical_proj = self.optical_proj(optical_feat)  # [B, C//4, H, W]
        terrain_proj = self.terrain_proj(terrain_feat)  # [B, C//4, H, W]
        
        # 计算注意力权重（地形指导光学）
        terrain_attention = self.attention_terrain(terrain_proj)  # [B, 1, H, W]
        terrain_attention = torch.sigmoid(terrain_attention)
        
        # 计算注意力权重（光学指导地形）
        optical_attention = self.attention_optical(optical_proj)  # [B, 1, H, W]
        optical_attention = torch.sigmoid(optical_attention)
        
        # 跨模态信息注入
        # 将投影的特征与原特征拼接
        optical_with_terrain = torch.cat([
            optical_feat, 
            terrain_proj * optical_attention  # 光学指导的地形特征
        ], dim=1)
        
        terrain_with_optical = torch.cat([
            terrain_feat,
            optical_proj * terrain_attention  # 地形指导的光学特征
        ], dim=1)
        
        # 融合
        optical_fused = self.fusion_optical(optical_with_terrain)
        terrain_fused = self.fusion_terrain(terrain_with_optical)
        
        return optical_fused, terrain_fused


class AdaptiveGateFusion(nn.Module):
    """
    自适应门控融合模块
    根据输入特征动态计算各模态的融合权重
    """
    def __init__(self, channels):
        super(AdaptiveGateFusion, self).__init__()
        
        # 门控权重生成器
        self.gate_generator = nn.Sequential(
            nn.Conv2d(channels * 4, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 4, 1),  # 4个权重：光学、地形、交叉光学、交叉地形
            nn.Softmax(dim=1)  # 权重归一化
        )
        
    def forward(self, optical, terrain, cross_optical, cross_terrain):
        """
        前向传播
        Args:
            optical: 原始光学特征
            terrain: 原始地形特征
            cross_optical: 交叉注意力后的光学特征
            cross_terrain: 交叉注意力后的地形特征
        Returns:
            融合后的特征
        """
        # 拼接所有特征
        all_feats = torch.cat([optical, terrain, cross_optical, cross_terrain], dim=1)
        
        # 生成门控权重 [B, 4, H, W]
        gates = self.gate_generator(all_feats)
        
        # 分离权重
        gate_optical = gates[:, 0:1, :, :]  # [B, 1, H, W]
        gate_terrain = gates[:, 1:2, :, :]
        gate_cross_optical = gates[:, 2:3, :, :]
        gate_cross_terrain = gates[:, 3:4, :, :]
        
        # 加权融合
        fused = (optical * gate_optical + 
                 terrain * gate_terrain + 
                 cross_optical * gate_cross_optical + 
                 cross_terrain * gate_cross_terrain)
        
        return fused


# ==================== 核心融合模块 ====================

class CrossScaleCrossModalFusion(nn.Module):
    """
    跨尺度跨模态融合模块 (Cross-Scale Cross-Modal Fusion Module)
    
    三个关键步骤：
    1. 模态内自注意力：提炼各自模态的重要特征
    2. 跨模态交叉注意力：让两个模态"对话"，实现特征互补
    3. 自适应门控融合：动态调整各模态的贡献权重
    """
    def __init__(self, channels):
        super(CrossScaleCrossModalFusion, self).__init__()
        
        # 1. 模态内自注意力 (Intra-Modal Self-Attention)
        self.optical_self_attn = ChannelSpatialAttention(channels)
        self.terrain_self_attn = ChannelSpatialAttention(channels)
        
        # 2. 跨模态交叉注意力 (Cross-Modal Attention)
        self.cross_attention = CrossModalAttention(channels)
        
        # 3. 自适应门控融合 (Adaptive Gated Fusion)
        self.gate_fusion = AdaptiveGateFusion(channels)
        
        # 4. 特征细化
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, optical_feat, terrain_feat):
        """
        前向传播
        Args:
            optical_feat: 光学特征 [B, C, H, W]
            terrain_feat: 地形特征 [B, C, H, W]
        Returns:
            fused_feat: 融合后的特征 [B, C, H, W]
        """
        # 步骤1: 模态内自注意力 (提炼各自特征)
        optical_refined = self.optical_self_attn(optical_feat)
        terrain_refined = self.terrain_self_attn(terrain_feat)
        
        # 步骤2: 跨模态交叉注意力 (模态间信息交换)
        cross_optical, cross_terrain = self.cross_attention(
            optical_refined, terrain_refined
        )
        
        # 步骤3: 自适应门控融合
        fused_feat = self.gate_fusion(
            optical_refined, terrain_refined,
            cross_optical, cross_terrain
        )
        
        # 步骤4: 特征细化
        fused_feat = self.refine(fused_feat)
        
        return fused_feat


# ==================== 主网络 ====================

class DSCM_Net(nn.Module):
    """
    双流跨模态特征融合网络 (Dual-Stream Cross-Modal Fusion Network)
    专为Landslide4Sense多源遥感数据设计
    
    输入:
        - 光学影像: 13通道 (Sentinel-2波段)
        - 地形数据: 3通道 (DEM, 坡度, 坡向)
    输出:
        - 滑坡分割图: 1通道 (二值分割)
    """
    def __init__(self, n_classes=1, optical_channels=13, terrain_channels=3):
        super(DSCM_Net, self).__init__()
        
        # 基础通道数
        base_channels = 64
        
        # ===== 1. 双流编码器初始卷积 =====
        # 光学流初始卷积 (处理13个波段)
        self.optical_conv1 = DoubleConv(optical_channels, base_channels)
        # 地形流初始卷积 (处理3个地形特征)
        self.terrain_conv1 = DoubleConv(terrain_channels, base_channels)
        
        # ===== 2. 下采样层 =====
        # 光学流下采样
        self.optical_down1 = Down(base_channels, base_channels * 2)
        self.optical_down2 = Down(base_channels * 2, base_channels * 4)
        self.optical_down3 = Down(base_channels * 4, base_channels * 8)
        self.optical_down4 = Down(base_channels * 8, base_channels * 16)
        
        # 地形流下采样
        self.terrain_down1 = Down(base_channels, base_channels * 2)
        self.terrain_down2 = Down(base_channels * 2, base_channels * 4)
        self.terrain_down3 = Down(base_channels * 4, base_channels * 8)
        self.terrain_down4 = Down(base_channels * 8, base_channels * 16)
        
        # ===== 3. 跨尺度跨模态融合模块 =====
        self.cs_cmf1 = CrossScaleCrossModalFusion(base_channels)        # 第1层融合
        self.cs_cmf2 = CrossScaleCrossModalFusion(base_channels * 2)    # 第2层融合
        self.cs_cmf3 = CrossScaleCrossModalFusion(base_channels * 4)    # 第3层融合
        self.cs_cmf4 = CrossScaleCrossModalFusion(base_channels * 8)    # 第4层融合
        
        # ===== 4. 共享解码器 =====
        self.up1 = Up(base_channels * 16, base_channels * 8, bilinear=True)
        self.up2 = Up(base_channels * 8, base_channels * 4, bilinear=True)
        self.up3 = Up(base_channels * 4, base_channels * 2, bilinear=True)
        self.up4 = Up(base_channels * 2, base_channels, bilinear=True)
        
        # ===== 5. 输出层 =====
        self.outc = nn.Conv2d(base_channels, n_classes, kernel_size=1)
        
        # ===== 6. 跳跃连接特征细化 =====
        # 用于调整跳跃连接特征的通道数
        self.skip_conv1 = nn.Conv2d(base_channels, base_channels, 1)
        self.skip_conv2 = nn.Conv2d(base_channels * 2, base_channels * 2, 1)
        self.skip_conv3 = nn.Conv2d(base_channels * 4, base_channels * 4, 1)
        self.skip_conv4 = nn.Conv2d(base_channels * 8, base_channels * 8, 1)
        
        # ===== 7. 瓶颈层融合卷积 =====
        self.bottleneck_fusion = nn.Sequential(
            nn.Conv2d(base_channels * 32, base_channels * 16, 3, padding=1),
            nn.BatchNorm2d(base_channels * 16),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, optical_input, terrain_input):
        """
        前向传播
        
        Args:
            optical_input: 光学影像 [B, 13, H, W]
            terrain_input: 地形数据 [B, 3, H, W]
        
        Returns:
            output: 分割结果 [B, 1, H, W]
        """
        
        # ===== 编码器阶段 1: 最高分辨率 =====
        # 双流特征提取
        optical_x1 = self.optical_conv1(optical_input)  # [B, 64, H, W]
        terrain_x1 = self.terrain_conv1(terrain_input)  # [B, 64, H, W]
        
        # 第1次跨模态融合
        fused_x1 = self.cs_cmf1(optical_x1, terrain_x1)  # [B, 64, H, W]
        skip1 = self.skip_conv1(fused_x1)  # 跳跃连接特征
        
        # ===== 编码器阶段 2: 1/2分辨率 =====
        optical_x2 = self.optical_down1(optical_x1)  # [B, 128, H/2, W/2]
        terrain_x2 = self.terrain_down1(terrain_x1)  # [B, 128, H/2, W/2]
        fused_x2 = self.cs_cmf2(optical_x2, terrain_x2)  # [B, 128, H/2, W/2]
        skip2 = self.skip_conv2(fused_x2)  # 跳跃连接特征
        
        # ===== 编码器阶段 3: 1/4分辨率 =====
        optical_x3 = self.optical_down2(optical_x2)  # [B, 256, H/4, W/4]
        terrain_x3 = self.terrain_down2(terrain_x2)  # [B, 256, H/4, W/4]
        fused_x3 = self.cs_cmf3(optical_x3, terrain_x3)  # [B, 256, H/4, W/4]
        skip3 = self.skip_conv3(fused_x3)  # 跳跃连接特征
        
        # ===== 编码器阶段 4: 1/8分辨率 =====
        optical_x4 = self.optical_down3(optical_x3)  # [B, 512, H/8, W/8]
        terrain_x4 = self.terrain_down3(terrain_x3)  # [B, 512, H/8, W/8]
        fused_x4 = self.cs_cmf4(optical_x4, terrain_x4)  # [B, 512, H/8, W/8]
        skip4 = self.skip_conv4(fused_x4)  # 跳跃连接特征
        
        # ===== 瓶颈层: 1/16分辨率 =====
        optical_x5 = self.optical_down4(optical_x4)  # [B, 1024, H/16, W/16]
        terrain_x5 = self.terrain_down4(terrain_x4)  # [B, 1024, H/16, W/16]
        
        # 瓶颈层融合 (简单拼接+卷积)
        bottleneck = torch.cat([optical_x5, terrain_x5], dim=1)  # [B, 2048, H/16, W/16]
        bottleneck = self.bottleneck_fusion(bottleneck)  # [B, 1024, H/16, W/16]
        
        # ===== 解码器阶段 =====
        # 第1次上采样
        x = self.up1(bottleneck, skip4)  # [B, 512, H/8, W/8]
        
        # 第2次上采样
        x = self.up2(x, skip3)  # [B, 256, H/4, W/4]
        
        # 第3次上采样
        x = self.up3(x, skip2)  # [B, 128, H/2, W/2]
        
        # 第4次上采样
        x = self.up4(x, skip1)  # [B, 64, H, W]
        
        # ===== 输出 =====
        output = self.outc(x)  # [B, 1, H, W]
        
        return output


# ==================== 简化版本（可选） ====================

class SimplifiedDSCM_Net(nn.Module):
    """
    简化版本的DSCM-Net
    减少了通道数和模块复杂度，适合快速实验
    """
    def __init__(self, n_classes=1, optical_channels=13, terrain_channels=3):
        super(SimplifiedDSCM_Net, self).__init__()
        
        # 基础通道数（减半）
        base_channels = 32
        
        # 光学流编码器
        self.optical_encoder = nn.Sequential(
            DoubleConv(optical_channels, base_channels),
            Down(base_channels, base_channels * 2),
            Down(base_channels * 2, base_channels * 4),
            Down(base_channels * 4, base_channels * 8),
            Down(base_channels * 8, base_channels * 16)
        )
        
        # 地形流编码器
        self.terrain_encoder = nn.Sequential(
            DoubleConv(terrain_channels, base_channels),
            Down(base_channels, base_channels * 2),
            Down(base_channels * 2, base_channels * 4),
            Down(base_channels * 4, base_channels * 8),
            Down(base_channels * 8, base_channels * 16)
        )
        
        # 跨模态融合模块（简化版）
        self.fusion1 = CrossScaleCrossModalFusion(base_channels)
        self.fusion2 = CrossScaleCrossModalFusion(base_channels * 2)
        self.fusion3 = CrossScaleCrossModalFusion(base_channels * 4)
        self.fusion4 = CrossScaleCrossModalFusion(base_channels * 8)
        
        # 解码器
        self.up1 = Up(base_channels * 32, base_channels * 8, bilinear=True)
        self.up2 = Up(base_channels * 8, base_channels * 4, bilinear=True)
        self.up3 = Up(base_channels * 4, base_channels * 2, bilinear=True)
        self.up4 = Up(base_channels * 2, base_channels, bilinear=True)
        
        # 输出层
        self.outc = nn.Conv2d(base_channels, n_classes, kernel_size=1)
    
    def forward(self, optical_input, terrain_input):
        # 编码器
        optical_features = []
        terrain_features = []
        
        # 第1层
        x = optical_input
        for layer in self.optical_encoder:
            x = layer(x)
            optical_features.append(x)
        
        x = terrain_input
        for layer in self.terrain_encoder:
            x = layer(x)
            terrain_features.append(x)
        
        # 融合
        fused1 = self.fusion1(optical_features[0], terrain_features[0])
        fused2 = self.fusion2(optical_features[1], terrain_features[1])
        fused3 = self.fusion3(optical_features[2], terrain_features[2])
        fused4 = self.fusion4(optical_features[3], terrain_features[3])
        
        # 瓶颈层融合
        bottleneck = torch.cat([optical_features[4], terrain_features[4]], dim=1)
        
        # 解码器
        x = self.up1(bottleneck, fused4)
        x = self.up2(x, fused3)
        x = self.up3(x, fused2)
        x = self.up4(x, fused1)
        
        # 输出
        output = self.outc(x)
        
        return output


# ==================== 测试代码 ====================

def test_dscm_net():
    """测试DSCM-Net模型"""
    print("=" * 50)
    print("测试DSCM-Net模型")
    print("=" * 50)
    
    # 创建模型
    model = DSCM_Net(
        n_classes=1,
        optical_channels=13,  # Sentinel-2的13个波段
        terrain_channels=3    # DEM, 坡度, 坡向
    )
    
    # 打印模型结构
    print(f"模型总参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print()
    
    # 创建模拟输入数据
    batch_size = 2
    height, width = 256, 256
    
    optical_input = torch.randn(batch_size, 13, height, width)
    terrain_input = torch.randn(batch_size, 3, height, width)
    
    print(f"光学输入尺寸: {optical_input.shape}")
    print(f"地形输入尺寸: {terrain_input.shape}")
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        output = model(optical_input, terrain_input)
    
    print(f"输出尺寸: {output.shape}")
    
    # 测试简化版本
    print("\n" + "=" * 50)
    print("测试简化版本SimplifiedDSCM_Net")
    print("=" * 50)
    
    simple_model = SimplifiedDSCM_Net(
        n_classes=1,
        optical_channels=13,
        terrain_channels=3
    )
    
    print(f"简化版模型总参数量: {sum(p.numel() for p in simple_model.parameters())/1e6:.2f}M")
    
    with torch.no_grad():
        simple_output = simple_model(optical_input, terrain_input)
    
    print(f"简化版输出尺寸: {simple_output.shape}")
    
    print("\n测试完成!")
    return model, simple_model


# ==================== 可视化函数 ====================

def visualize_attention_weights(model, optical_input, terrain_input):
    """
    可视化注意力权重（用于模型分析）
    """
    import matplotlib.pyplot as plt
    
    # 设置为评估模式
    model.eval()
    
    # 前向传播并获取中间特征
    # 注意：这需要修改模型以返回中间特征
    
    print("注意力可视化功能需要修改模型以返回中间特征")
    print("可以在forward方法中添加hook或直接返回中间结果")


# ==================== 训练辅助函数 ====================

def get_dscm_net_loss_function():
    """
    获取适合DSCM-Net的损失函数
    推荐使用Dice Loss + BCE Loss的组合
    """
    def dice_loss(pred, target, smooth=1e-5):
        """Dice Loss"""
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()
    
    def bce_loss(pred, target):
        """Binary Cross Entropy Loss"""
        return F.binary_cross_entropy_with_logits(pred, target)
    
    def combined_loss(pred, target, alpha=0.5):
        """组合损失函数"""
        dice = dice_loss(pred, target)
        bce = bce_loss(pred, target)
        return alpha * dice + (1 - alpha) * bce
    
    return combined_loss


# ==================== 主程序 ====================

if __name__ == "__main__":
    # 测试模型
    model, simple_model = test_dscm_net()
    
    # 使用示例
    print("\n" + "=" * 50)
    print("使用示例")
    print("=" * 50)
    
    print("""
    # 1. 创建模型
    model = DSCM_Net(
        n_classes=1,
        optical_channels=13,
        terrain_channels=3
    )
    
    # 2. 准备数据
    # optical_data: [B, 13, H, W] - Sentinel-2多光谱数据
    # terrain_data: [B, 3, H, W] - DEM及派生特征
    
    # 3. 前向传播
    output = model(optical_data, terrain_data)
    
    # 4. 计算损失
    loss_fn = get_dscm_net_loss_function()
    loss = loss_fn(output, target_mask)
    
    # 5. 训练优化
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    """)