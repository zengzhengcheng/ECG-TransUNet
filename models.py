import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# 1. 基础组件 (Blocks)
# ==========================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block: 学习通道重要性"""

    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class ResBlock(nn.Module):
    """带 SE 注意力的残差块"""

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.GELU()  # 使用 GELU
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SEBlock(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class BiLSTM_Block(nn.Module):
    """
    【核心时序模块】
    在 U-Net 瓶颈层加入双向 LSTM，提取全局时序节奏（心率间隔）。
    """

    def __init__(self, in_channels, hidden_size):
        super(BiLSTM_Block, self).__init__()
        # PyTorch LSTM 输入: (Batch, Seq_Len, Features)
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        # 映射回原通道 (hidden * 2 -> original channel)
        self.out_conv = nn.Sequential(
            nn.Linear(hidden_size * 2, in_channels),
            nn.GELU()
        )

    def forward(self, x):
        # x shape: (B, C, L)
        b, c, l = x.size()

        # 1. 维度置换: CNN(B, C, L) -> LSTM(B, L, C)
        x_permuted = x.permute(0, 2, 1)

        # 2. LSTM 前向
        lstm_out, _ = self.lstm(x_permuted)

        # 3. 映射回原通道
        out = self.out_conv(lstm_out)

        # 4. 维度置换回: (B, L, C) -> (B, C, L)
        out = out.permute(0, 2, 1)

        # 5. 残差连接
        return x + out


# ==========================================
# 2. 分割模型 (BiLSTM + SE-ResUNet)
# ==========================================

class ECG_SE_ResUNet(nn.Module):
    """
    带有 Bi-LSTM 和 SE-Block 的双通道输入 U-Net
    """

    def __init__(self, in_channels=3, n_classes=1):  # 默认为2 (ECG + Initial Guess)
        super(ECG_SE_ResUNet, self).__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(32), nn.GELU()
        )

        # Encoder
        self.enc1 = self._make_layer(32, 32)
        self.pool1 = nn.MaxPool1d(2)

        self.enc2 = self._make_layer(32, 64)
        self.pool2 = nn.MaxPool1d(2)

        self.enc3 = self._make_layer(64, 128)
        self.pool3 = nn.MaxPool1d(2)

        self.enc4 = self._make_layer(128, 256)
        self.pool4 = nn.MaxPool1d(2)

        # Bottleneck (CNN + Bi-LSTM)
        self.bottleneck_conv = self._make_layer(256, 512)
        # 这里嵌入了 BiLSTM 模块
        self.bottleneck_lstm = BiLSTM_Block(in_channels=512, hidden_size=256)

        # Decoder
        self.up4 = nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2)
        self.dec4 = self._make_layer(512, 256)

        self.up3 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self._make_layer(256, 128)

        self.up2 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self._make_layer(128, 64)

        self.up1 = nn.ConvTranspose1d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self._make_layer(64, 32)

        self.out_conv = nn.Conv1d(32, n_classes, kernel_size=1)

    def _make_layer(self, in_c, out_c):
        ds = None
        if in_c != out_c:
            ds = nn.Sequential(nn.Conv1d(in_c, out_c, kernel_size=1, bias=False), nn.BatchNorm1d(out_c))
        return nn.Sequential(ResBlock(in_c, out_c, downsample=ds), ResBlock(out_c, out_c))

    def forward(self, x):
        # Encoder
        x = self.stem(x)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Bottleneck
        b = self.bottleneck_conv(self.pool4(e4))
        b = self.bottleneck_lstm(b)  # 经过 LSTM 修正时序特征

        # Decoder
        d4 = self.dec4(torch.cat([e4, self.up4(b)], dim=1))
        d3 = self.dec3(torch.cat([e3, self.up3(d4)], dim=1))
        d2 = self.dec2(torch.cat([e2, self.up2(d3)], dim=1))
        d1 = self.dec1(torch.cat([e1, self.up1(d2)], dim=1))

        return self.out_conv(d1)


# ==========================================
# 3. 质量评估模型 (ResNet-34)
# ==========================================

class ECG_ResNet34_SQA(nn.Module):
    def __init__(self, in_channels=1):
        super(ECG_ResNet34_SQA, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.GELU()
        self.maxpool = nn.MaxPool1d(3, 2, 1)

        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 1)
        )

    def _make_layer(self, planes, blocks, stride=1):
        ds = None
        if stride != 1 or self.inplanes != planes:
            ds = nn.Sequential(nn.Conv1d(self.inplanes, planes, 1, stride, bias=False), nn.BatchNorm1d(planes))
        layers = [ResBlock(self.inplanes, planes, stride, ds)]
        self.inplanes = planes
        for _ in range(1, blocks): layers.append(ResBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(self.avgpool(x).flatten(1))


# ==========================================
# 4. 模型注册表 (Registry)
# ==========================================

class ECGModelRegistry:
    """
    负责所有模型的生命周期管理
    """

    def __init__(self, in_channels=3,device='cpu'):
        self.device = torch.device(device)
        self.seg_model = None
        self._init_models(in_channels=in_channels)

    def _init_models(self):
        # 初始化 Seg 模型 (双通道，带 BiLSTM)
        self.seg_model = ECG_SE_ResUNet(in_channels=3, n_classes=1).to(self.device)

    def reset_models(self):
        self._init_models()
        print("[Registry] Models reset to random initialization.")

    def get_optimizers(self, lr_seg=1e-3, lr_sqa=1e-3):
        return torch.optim.Adam(self.seg_model.parameters(), lr=lr_seg)


    def export_state_bundle(self):
        return {
            'seg_model_state': self.seg_model.state_dict(),
            'version': 'BiLSTM_ResUNet_Final'
        }

    def load_state_bundle(self, bundle):
        try:
            self.seg_model.load_state_dict(bundle['seg_model_state'])
            return True
        except Exception as e:
            print(f"[Registry] Load failed: {e}")
            return False

    def to_eval_mode(self):
        self.seg_model.eval()

    def to_train_mode(self):
        self.seg_model.train()
