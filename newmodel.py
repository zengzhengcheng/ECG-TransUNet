import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# 1. 高级组件 (Attention & Multi-scale)
# ==========================================

class ChannelAttention(nn.Module):
    """CBAM: 通道注意力"""

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return torch.sigmoid(out)


class SpatialAttention(nn.Module):
    """CBAM: 空间(时间)注意力 - 关注 R 峰位置"""

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        # 在通道维度上做 max 和 avg
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return torch.sigmoid(out)


class CBAMBlock(nn.Module):
    """结合通道和空间注意力"""

    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out


class MultiScaleResBlock(nn.Module):
    """
    多尺度残差块 (类似 Inception)。
    并行使用不同大小的卷积核来捕捉高频(R波)和低频(T波)特征。
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(MultiScaleResBlock, self).__init__()

        # 主干路径分为三路：小核(3), 中核(11), 大核(21)
        self.scale1 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=3, stride=stride, padding=1)
        self.scale2 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=11, stride=stride, padding=5)
        self.scale3 = nn.Conv1d(in_channels, out_channels // 2, kernel_size=21, stride=stride, padding=10)

        # 融合后的处理
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.GELU()

        # 注意力机制
        self.cbam = CBAMBlock(out_channels)

        # 快捷连接 (Shortcut)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        # 多尺度特征提取
        s1 = self.scale1(x)
        s2 = self.scale2(x)
        s3 = self.scale3(x)

        # 拼接
        out = torch.cat([s1, s2, s3], dim=1)
        out = self.bn(out)
        out = self.relu(out)

        # 加注意力
        out = self.cbam(out)

        # 残差连接
        out += self.shortcut(x)
        return self.relu(out)


# ==========================================
# 2. 超级骨干网络 (Advanced UNet)
# ==========================================

class AdvancedECGUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(AdvancedECGUNet, self).__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(32), nn.GELU()
        )

        # Encoder (Multi-Scale + CBAM)
        self.enc1 = MultiScaleResBlock(32, 64)
        self.pool1 = nn.MaxPool1d(2)

        self.enc2 = MultiScaleResBlock(64, 128)
        self.pool2 = nn.MaxPool1d(2)

        self.enc3 = MultiScaleResBlock(128, 256)
        self.pool3 = nn.MaxPool1d(2)

        self.enc4 = MultiScaleResBlock(256, 512)
        self.pool4 = nn.MaxPool1d(2)

        # Bottleneck (Bi-LSTM)
        # 将 LSTM 放在最深处，处理时序依赖
        self.bottleneck_conv = MultiScaleResBlock(512, 1024)
        self.lstm = nn.LSTM(1024, 512, num_layers=2, batch_first=True, bidirectional=True)
        self.lstm_proj = nn.Linear(1024, 1024)

        # Decoder
        self.up4 = nn.ConvTranspose1d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = MultiScaleResBlock(1024, 512)  # 512+512

        self.up3 = nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2)
        self.dec3 = MultiScaleResBlock(512, 256)

        self.up2 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.dec2 = MultiScaleResBlock(256, 128)

        self.up1 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.dec1 = MultiScaleResBlock(128, 64)

        # Output Head
        self.out_conv = nn.Conv1d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x = self.stem(x)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Bottleneck
        b = self.bottleneck_conv(self.pool4(e4))  # (B, 1024, L/16)

        # LSTM 处理
        # permute -> (B, L, C)
        b_perm = b.permute(0, 2, 1)
        lstm_out, _ = self.lstm(b_perm)
        b_rec = self.lstm_proj(lstm_out).permute(0, 2, 1)  # back to (B, C, L)
        b = b + b_rec  # 残差连接

        # Decoder
        d4 = self.dec4(torch.cat([e4, self.up4(b)], dim=1))
        d3 = self.dec3(torch.cat([e3, self.up3(d4)], dim=1))
        d2 = self.dec2(torch.cat([e2, self.up2(d3)], dim=1))
        d1 = self.dec1(torch.cat([e1, self.up1(d2)], dim=1))

        return self.out_conv(d1)