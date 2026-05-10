import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==================== 基础模块 (与U-Net兼容) ====================

class DoubleConv(nn.Module):
    """双重卷积模块"""
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


# ==================== Swin Transformer相关模块 ====================

class WindowAttention(nn.Module):
    """窗口多头自注意力 (W-MSA)"""
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 相对位置偏置表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        
        # 生成相对位置索引
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # 偏移到从0开始
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B_, num_heads, N, head_dim]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer块"""
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        assert 0 <= self.shift_size < self.window_size, "shift_size必须在0到window_size之间"

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
        
        # 生成注意力掩码
        if self.shift_size > 0:
            H, W = 56, 56  # 假设输入尺寸，实际会动态调整
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                       slice(-self.window_size, -self.shift_size),
                       slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                       slice(-self.window_size, -self.shift_size),
                       slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "输入特征长度必须等于H*W"
        
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        # 循环移位
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        
        # 划分窗口
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        
        # 合并窗口
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
        
        # 反向循环移位
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        
        x = x.view(B, H * W, C)
        x = shortcut + x
        
        # FFN
        x = x + self.mlp(self.norm2(x))
        
        return x


def window_partition(x, window_size):
    """
    将特征图划分为不重叠的窗口
    Args:
        x: (B, H, W, C)
        window_size (int): 窗口大小
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    将窗口合并回特征图
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): 窗口大小
        H (int): 特征图高度
        W (int): 特征图宽度
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# ==================== 通道-空间注意力模块 ====================

class ChannelSpatialAttention(nn.Module):
    """通道-空间协同注意力模块"""
    def __init__(self, channels, reduction=16):
        super(ChannelSpatialAttention, self).__init__()
        
        # 通道注意力
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


# ==================== 多尺度上下文聚合瓶颈 ====================

class MultiScaleContextAggregationBottleneck(nn.Module):
    """多尺度上下文聚合瓶颈 (MCAB)"""
    def __init__(self, in_channels, out_channels, dilation_rates=[1, 3, 6, 12]):
        super(MultiScaleContextAggregationBottleneck, self).__init__()
        
        # 多个并行分支，不同的空洞率
        self.branches = nn.ModuleList()
        for rate in dilation_rates:
            branch = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate),
                ChannelSpatialAttention(out_channels),  # 通道-空间注意力
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.branches.append(branch)
        
        # 全局平均池化分支
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 融合层
        total_channels = out_channels * (len(dilation_rates) + 1)
        self.fusion = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """前向传播"""
        branch_outputs = []
        
        # 并行分支
        for branch in self.branches:
            branch_out = branch(x)
            branch_outputs.append(branch_out)
        
        # 全局分支
        global_out = self.global_branch(x)
        global_out = F.interpolate(global_out, size=x.shape[2:], mode='bilinear', align_corners=True)
        branch_outputs.append(global_out)
        
        # 拼接所有分支输出
        concatenated = torch.cat(branch_outputs, dim=1)
        
        # 融合
        output = self.fusion(concatenated)
        
        return output


# ==================== 细节增强模块 ====================

class DetailEnhancementModule(nn.Module):
    """细节增强模块 (用于强化跳跃连接特征)"""
    def __init__(self, in_channels, out_channels):
        super(DetailEnhancementModule, self).__init__()
        
        # 边缘检测分支
        self.edge_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        
        # 纹理增强分支
        self.texture_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, dilation=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 注意力门控
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 边缘特征
        edge_feat = self.edge_branch(x)
        
        # 纹理特征
        texture_feat = self.texture_branch(x)
        
        # 融合
        fused = torch.cat([edge_feat, texture_feat], dim=1)
        fused = self.fusion(fused)
        
        # 门控加权
        gate = self.gate(fused)
        enhanced = fused * gate + x  # 残差连接
        
        return enhanced


# ==================== 空间→序列投影层 ====================

class SpatialToSequenceProjection(nn.Module):
    """将CNN特征图转换为Transformer序列"""
    def __init__(self, in_channels, embed_dim, patch_size=4):
        super(SpatialToSequenceProjection, self).__init__()
        self.patch_size = patch_size
        
        # 投影层：将每个patch投影到embed_dim维度
        self.projection = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # 层归一化
        self.norm = nn.LayerNorm(embed_dim)
        
        # 位置编码 (可学习的)
        # 假设最大输入尺寸为224x224
        max_seq_len = (224 // patch_size) ** 2
        self.pos_embed = nn.Parameter(
            torch.zeros(1, max_seq_len, embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=.02)
    
    def forward(self, x):
        """
        输入: x [B, C, H, W]
        输出: seq [B, N, D], N = (H/patch_size)*(W/patch_size)
        """
        B, C, H, W = x.shape
        
        # 投影到嵌入空间
        x = self.projection(x)  # [B, D, H/p, W/p]
        
        # 展平为序列 [B, N, D]
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        
        # 层归一化
        x = self.norm(x)
        
        # 添加位置编码
        N = x.shape[1]
        if N <= self.pos_embed.shape[1]:
            pos_embed = self.pos_embed[:, :N, :]
        else:
            # 插值以适应更大的序列长度
            pos_embed = F.interpolate(
                self.pos_embed.transpose(1, 2),
                size=N,
                mode='linear'
            ).transpose(1, 2)
        
        x = x + pos_embed
        
        return x


# ==================== 渐进式混合编码器 ====================

class ProgressiveHybridEncoder(nn.Module):
    """渐进式混合编码器：浅层CNN + 深层Swin Transformer"""
    def __init__(self, in_channels=16, base_channels=64):
        super(ProgressiveHybridEncoder, self).__init__()
        
        # ===== 阶段1-2: CNN编码器 (高分辨率) =====
        # 阶段1: 原始分辨率 → 1/2分辨率
        self.cnn_stage1 = nn.Sequential(
            DoubleConv(in_channels, base_channels),
            nn.MaxPool2d(2)  # 下采样到 [H/2, W/2]
        )
        
        # 阶段2: 1/2分辨率 → 1/4分辨率
        self.cnn_stage2 = nn.Sequential(
            DoubleConv(base_channels, base_channels * 2),
            nn.MaxPool2d(2)  # 下采样到 [H/4, W/4]
        )
        
        # ===== 过渡层: CNN特征 → Transformer序列 =====
        self.transition = SpatialToSequenceProjection(
            in_channels=base_channels * 2,
            embed_dim=base_channels * 4,  # Transformer嵌入维度
            patch_size=4  # 将特征图划分为4x4的块
        )
        
        # ===== 阶段3: Swin Transformer编码器 (1/4分辨率) =====
        self.swin_stage3 = nn.ModuleList([
            SwinTransformerBlock(
                dim=base_channels * 4,
                num_heads=4,
                window_size=7,
                shift_size=0 if i % 2 == 0 else 7 // 2,  # 交替使用W-MSA和SW-MSA
                mlp_ratio=4.0
            )
            for i in range(2)  # 2个Swin Transformer块
        ])
        
        # ===== 阶段4: Swin Transformer编码器 (1/8分辨率) =====
        # 下采样层
        self.downsample_3to4 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, 3, stride=2, padding=1),
            nn.LayerNorm(base_channels * 8),
            nn.GELU()
        )
        
        # Swin Transformer阶段4
        self.swin_stage4 = nn.ModuleList([
            SwinTransformerBlock(
                dim=base_channels * 8,
                num_heads=8,
                window_size=7,
                shift_size=0 if i % 2 == 0 else 7 // 2,
                mlp_ratio=4.0
            )
            for i in range(2)  # 2个Swin Transformer块
        ])
        
        # 用于从序列转换回空间的特征
        self.seq_to_space_norm = nn.LayerNorm(base_channels * 4)
        self.seq_to_space_norm2 = nn.LayerNorm(base_channels * 8)
    
    def forward(self, x):
        """
        前向传播
        输入: x [B, C, H, W]
        输出: 各阶段特征字典
        """
        # ===== 阶段1: CNN (高分辨率) =====
        stage1_feat = self.cnn_stage1(x)  # [B, 64, H/2, W/2]
        
        # ===== 阶段2: CNN (中等分辨率) =====
        stage2_feat = self.cnn_stage2(stage1_feat)  # [B, 128, H/4, W/4]
        
        # ===== 过渡到Transformer =====
        B, C, H2, W2 = stage2_feat.shape
        stage2_seq = self.transition(stage2_feat)  # [B, N, 256], N=(H2/4)*(W2/4)
        
        # ===== 阶段3: Swin Transformer (1/4分辨率) =====
        # 将序列重塑回空间格式以获取H, W
        H3, W3 = H2 // 4, W2 // 4  # 因为patch_size=4
        
        # 应用Swin Transformer块
        x_seq = stage2_seq
        for blk in self.swin_stage3:
            x_seq = blk(x_seq, H3, W3)
        
        # 转换回空间特征
        stage3_seq = x_seq
        stage3_feat = self.seq_to_space_norm(stage3_seq)
        stage3_feat = stage3_feat.transpose(1, 2).view(B, -1, H3, W3)  # [B, 256, H/4, W/4]
        
        # ===== 阶段4: Swin Transformer (1/8分辨率) =====
        # 下采样
        stage3_feat_down = self.downsample_3to4(stage3_feat)  # [B, 512, H/8, W/8]
        H4, W4 = H3 // 2, W3 // 2
        
        # 转换为序列
        B, C4, H4, W4 = stage3_feat_down.shape
        stage3_seq_down = stage3_feat_down.flatten(2).transpose(1, 2)  # [B, N, 512]
        
        # 应用Swin Transformer块
        x_seq = stage3_seq_down
        for blk in self.swin_stage4:
            x_seq = blk(x_seq, H4, W4)
        
        # 转换回空间特征
        stage4_seq = x_seq
        stage4_feat = self.seq_to_space_norm2(stage4_seq)
        stage4_feat = stage4_feat.transpose(1, 2).view(B, -1, H4, W4)  # [B, 512, H/8, W/8]
        
        return {
            'stage1': stage1_feat,  # [B, 64, H/2, W/2]
            'stage2': stage2_feat,  # [B, 128, H/4, W/4]
            'stage3': stage3_feat,  # [B, 256, H/4, W/4]
            'stage4': stage4_feat,  # [B, 512, H/8, W/8]
        }


# ==================== 完整HSC-HENet模型 ====================

class HSC_HENet(nn.Module):
    """
    层次化Swin-CNN混合编码网络 (Hierarchical Swin-CNN Hybrid Encoder Network)
    专为滑坡检测任务设计
    
    特点:
    1. 渐进式混合编码器: 浅层CNN + 深层Swin Transformer
    2. 多尺度上下文聚合瓶颈 (MCAB)
    3. 细节增强的跳跃连接
    4. 深度监督训练
    """
    def __init__(self, n_channels=16, n_classes=1, base_channels=64, deep_supervision=False):
        super(HSC_HENet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.base_channels = base_channels
        self.deep_supervision = deep_supervision
        
        # ===== 1. 渐进式混合编码器 =====
        self.encoder = ProgressiveHybridEncoder(
            in_channels=n_channels,
            base_channels=base_channels
        )
        
        # ===== 2. 多尺度上下文聚合瓶颈 (MCAB) =====
        self.mcab = MultiScaleContextAggregationBottleneck(
            in_channels=base_channels * 8,  # 来自Swin阶段4的512通道
            out_channels=base_channels * 8,
            dilation_rates=[1, 3, 6, 12]
        )
        
        # ===== 3. 细节增强模块 (用于跳跃连接) =====
        self.detail_enhance1 = DetailEnhancementModule(base_channels, base_channels)  # 64 → 64
        self.detail_enhance2 = DetailEnhancementModule(base_channels * 2, base_channels * 2)  # 128 → 128
        self.detail_enhance3 = DetailEnhancementModule(base_channels * 4, base_channels * 4)  # 256 → 256
        
        # ===== 4. 解码器 =====
        self.up1 = Up(base_channels * 8, base_channels * 4, bilinear=True)  # 512 → 256
        self.up2 = Up(base_channels * 4, base_channels * 2, bilinear=True)  # 256 → 128
        self.up3 = Up(base_channels * 2, base_channels, bilinear=True)     # 128 → 64
        self.up4 = Up(base_channels, base_channels // 2, bilinear=True)    # 64 → 32
        
        # ===== 5. 输出层 =====
        self.outc = nn.Conv2d(base_channels // 2, n_classes, kernel_size=1)
        
        # ===== 6. 深度监督输出 (可选) =====
        if deep_supervision:
            self.aux_out1 = nn.Conv2d(base_channels * 4, n_classes, kernel_size=1)
            self.aux_out2 = nn.Conv2d(base_channels * 2, n_classes, kernel_size=1)
            self.aux_out3 = nn.Conv2d(base_channels, n_classes, kernel_size=1)
        
        # ===== 7. 特征细化卷积 =====
        self.refine_conv = nn.Sequential(
            nn.Conv2d(base_channels // 2, base_channels // 4, 3, padding=1),
            nn.BatchNorm2d(base_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels // 4, base_channels // 4, 3, padding=1),
            nn.BatchNorm2d(base_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        # ===== 8. 初始化参数 =====
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_aux=False):
        """
        前向传播
        
        参数:
            x: 输入图像 [B, 16, H, W] (13个光学波段 + 3个地形特征)
            return_aux: 是否返回辅助输出 (用于深度监督训练)
        
        返回:
            main_output: 主分割输出 [B, 1, H, W]
            aux_outputs: 辅助输出列表 (仅在return_aux=True且训练模式下返回)
        """
        # ===== 编码阶段 =====
        encoder_features = self.encoder(x)
        
        # 提取各层特征
        stage1_feat = encoder_features['stage1']  # [B, 64, H/2, W/2]
        stage2_feat = encoder_features['stage2']  # [B, 128, H/4, W/4]
        stage3_feat = encoder_features['stage3']  # [B, 256, H/4, W/4]
        stage4_feat = encoder_features['stage4']  # [B, 512, H/8, W/8]
        
        # ===== 多尺度上下文聚合 =====
        bottleneck = self.mcab(stage4_feat)  # [B, 512, H/8, W/8]
        
        # ===== 解码阶段 =====
        # 上采样1: 512 → 256 (需要stage3特征)
        up1 = self.up1(bottleneck, self.detail_enhance3(stage3_feat))  # [B, 256, H/4, W/4]
        
        # 上采样2: 256 → 128 (需要stage2特征)
        up2 = self.up2(up1, self.detail_enhance2(stage2_feat))  # [B, 128, H/2, W/2]
        
        # 上采样3: 128 → 64 (需要stage1特征)
        up3 = self.up3(up2, self.detail_enhance1(stage1_feat))  # [B, 64, H/2, W/2]
        
        # 上采样4: 64 → 32 (没有跳跃连接)
        up4 = self.up4(up3, None)  # [B, 32, H, W]
        
        # ===== 特征细化 =====
        refined = self.refine_conv(up4)  # [B, 16, H, W]
        
        # ===== 主输出 =====
        main_output = self.outc(refined)  # [B, 1, H, W]
        
        # ===== 深度监督输出 (仅在训练时) =====
        if return_aux and self.training and self.deep_supervision:
            # 辅助输出1: 来自解码器第1层
            aux1 = self.aux_out1(up1)  # [B, 1, H/4, W/4]
            aux1 = F.interpolate(aux1, size=x.shape[2:], mode='bilinear', align_corners=True)
            
            # 辅助输出2: 来自解码器第2层
            aux2 = self.aux_out2(up2)  # [B, 1, H/2, W/2]
            aux2 = F.interpolate(aux2, size=x.shape[2:], mode='bilinear', align_corners=True)
            
            # 辅助输出3: 来自解码器第3层
            aux3 = self.aux_out3(up3)  # [B, 1, H/2, W/2]
            aux3 = F.interpolate(aux3, size=x.shape[2:], mode='bilinear', align_corners=True)
            
            return main_output, [aux1, aux2, aux3]
        
        return main_output


# ==================== 简化版本 (可选) ====================

class SimplifiedHSC_HENet(nn.Module):
    """
    简化版本的HSC-HENet
    减少计算复杂度，适合快速实验
    """
    def __init__(self, n_channels=16, n_classes=1):
        super(SimplifiedHSC_HENet, self).__init__()
        
        # 减少基础通道数
        base_channels = 32
        
        # CNN编码器 (浅层)
        self.cnn_encoder = nn.Sequential(
            DoubleConv(n_channels, base_channels),
            Down(base_channels, base_channels * 2),
            Down(base_channels * 2, base_channels * 4)
        )
        
        # 过渡到Transformer
        self.transition = nn.Conv2d(base_channels * 4, base_channels * 8, 3, stride=2, padding=1)
        
        # 简化的Transformer层 (使用简单的自注意力)
        self.transformer_layer = nn.Sequential(
            nn.LayerNorm(base_channels * 8),
            nn.MultiheadAttention(base_channels * 8, num_heads=4, batch_first=True),
            nn.Linear(base_channels * 8, base_channels * 8),
            nn.GELU()
        )
        
        # 解码器
        self.up1 = Up(base_channels * 8, base_channels * 4, bilinear=True)
        self.up2 = Up(base_channels * 4, base_channels * 2, bilinear=True)
        self.up3 = Up(base_channels * 2, base_channels, bilinear=True)
        self.up4 = Up(base_channels, base_channels // 2, bilinear=True)
        
        # 输出
        self.outc = nn.Conv2d(base_channels // 2, n_classes, kernel_size=1)
        
        # 注意力模块 (简化)
        self.attention = ChannelSpatialAttention(base_channels * 4)
    
    def forward(self, x):
        # CNN编码
        cnn_features = []
        for layer in self.cnn_encoder:
            x = layer(x)
            cnn_features.append(x)
        
        # 过渡
        x = self.transition(x)
        B, C, H, W = x.shape
        
        # Transformer处理
        x_seq = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        x_seq, _ = self.transformer_layer[1](x_seq, x_seq, x_seq)  # 自注意力
        x = x_seq.transpose(1, 2).view(B, C, H, W)
        
        # 解码
        x = self.up1(x, self.attention(cnn_features[2]))
        x = self.up2(x, cnn_features[1])
        x = self.up3(x, cnn_features[0])
        x = self.up4(x, None)
        
        # 输出
        output = self.outc(x)
        
        return output


# ==================== 测试代码 ====================

def test_hsc_henet():
    """测试HSC-HENet模型"""
    print("=" * 60)
    print("测试HSC-HENet模型")
    print("=" * 60)
    
    # 创建模型
    model = HSC_HENet(
        n_channels=16,  # 13个光学波段 + 3个地形特征
        n_classes=1,
        base_channels=64,
        deep_supervision=True
    )
    
    # 打印模型信息
    print(f"模型总参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print(f"是否启用深度监督: {model.deep_supervision}")
    print()
    
    # 创建模拟输入数据
    batch_size = 2
    height, width = 256, 256
    
    input_tensor = torch.randn(batch_size, 16, height, width)
    
    print(f"输入尺寸: {input_tensor.shape}")
    
    # 测试训练模式
    print("\n1. 训练模式 (带深度监督):")
    model.train()
    with torch.no_grad():
        main_output, aux_outputs = model(input_tensor, return_aux=True)
    
    print(f"主输出尺寸: {main_output.shape}")
    for i, aux in enumerate(aux_outputs):
        print(f"辅助输出{i+1}尺寸: {aux.shape}")
    
    # 测试评估模式
    print("\n2. 评估模式 (无深度监督):")
    model.eval()
    with torch.no_grad():
        eval_output = model(input_tensor, return_aux=False)
    
    print(f"评估输出尺寸: {eval_output.shape}")
    
    # 测试简化版本
    print("\n" + "=" * 60)
    print("测试简化版本SimplifiedHSC_HENet")
    print("=" * 60)
    
    simple_model = SimplifiedHSC_HENet(
        n_channels=16,
        n_classes=1
    )
    
    print(f"简化版模型总参数量: {sum(p.numel() for p in simple_model.parameters())/1e6:.2f}M")
    
    with torch.no_grad():
        simple_output = simple_model(input_tensor)
    
    print(f"简化版输出尺寸: {simple_output.shape}")
    
    print("\n测试完成!")
    return model, simple_model


# ==================== 训练辅助函数 ====================

def get_hsc_henet_loss_function(deep_supervision=False, aux_weight=0.3):
    """
    获取适合HSC-HENet的损失函数
    
    参数:
        deep_supervision: 是否使用深度监督
        aux_weight: 辅助损失权重
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
    
    def combined_loss(pred, target, alpha=0.7):
        """组合损失函数 (Dice + BCE)"""
        dice = dice_loss(pred, target)
        bce = bce_loss(pred, target)
        return alpha * dice + (1 - alpha) * bce
    
    if deep_supervision:
        def deep_supervision_loss(preds, target):
            """深度监督损失函数"""
            main_pred, aux_preds = preds
            main_loss = combined_loss(main_pred, target)
            
            aux_loss = 0
            for aux_pred in aux_preds:
                aux_loss += combined_loss(aux_pred, target) * aux_weight
            
            return main_loss + aux_loss
        
        return deep_supervision_loss
    else:
        return combined_loss


# ==================== 主程序 ====================

if __name__ == "__main__":
    # 测试模型
    print("开始测试HSC-HENet模型...")
    model, simple_model = test_hsc_henet()
    
    # 使用示例
    print("\n" + "=" * 60)
    print("使用示例")
    print("=" * 60)
    
    print("""
    # 1. 创建模型 (带深度监督)
    model = HSC_HENet(
        n_channels=16,  # 13个光学波段 + 3个地形特征
        n_classes=1,
        base_channels=64,
        deep_supervision=True  # 训练时使用深度监督
    )
    
    # 2. 准备数据
    # input_data: [B, 16, H, W] - 多源遥感数据
    
    # 3. 训练阶段 (深度监督)
    model.train()
    main_output, aux_outputs = model(input_data, return_aux=True)
    loss_fn = get_hsc_henet_loss_function(deep_supervision=True)
    loss = loss_fn((main_output, aux_outputs), target_mask)
    
    # 4. 评估阶段
    model.eval()
    with torch.no_grad():
        output = model(input_data, return_aux=False)
    
    # 5. 训练优化
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    # 6. 渐进式训练策略
    #   阶段1: 主要训练CNN部分 (前10轮)
    #   阶段2: 联合训练CNN和Transformer (10-20轮)
    #   阶段3: 微调所有参数 (20轮后)
    """)
    
    # 参数统计
    print("\n模型参数统计:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")