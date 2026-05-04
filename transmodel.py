import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==========================================
# 1. 基础组件 (ResNet + Attention)
# ==========================================

class SEBlock(nn.Module):
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
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.GELU()
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


# ==========================================
# 2. Transformer 组件 (核心算力消耗点)
# ==========================================

class PositionalEncoding(nn.Module):
    """为 Transformer 提供位置信息"""

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(1, 2)  # (1, C, L)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (Batch, Channels, Len)
        return x + self.pe[:, :, :x.size(2)]


class TransformerBottleneck(nn.Module):
    """
    1D Transformer Encoder Block.
    输入: (B, C, L) -> 输出: (B, C, L)
    """

    def __init__(self, d_model, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        super(TransformerBottleneck, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.d_model = d_model

    def forward(self, src):
        # src: (Batch, Channels, Length)
        src = self.pos_encoder(src)

        # Transformer 需要 (Batch, Length, Channels)
        src = src.permute(0, 2, 1)

        output = self.transformer_encoder(src)

        # 变回 (Batch, Channels, Length)
        output = output.permute(0, 2, 1)
        return output


# ==========================================
# 3. TransUNet (CNN Encoder -> Transformer -> CNN Decoder)
# ==========================================

class ECG_TransUNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1):
        super(ECG_TransUNet, self).__init__()

        # --- CNN Encoder (负责下采样和局部特征) ---
        # Stem: 15360 -> 15360
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(64), nn.GELU()
        )

        # Layer 1: 15360 -> 7680
        self.enc1 = self._make_layer(64, 64, blocks=2)
        self.pool1 = nn.MaxPool1d(2)

        # Layer 2: 7680 -> 3840
        self.enc2 = self._make_layer(64, 128, blocks=2)
        self.pool2 = nn.MaxPool1d(2)

        # Layer 3: 3840 -> 1920
        self.enc3 = self._make_layer(128, 256, blocks=3)
        self.pool3 = nn.MaxPool1d(2)

        # Layer 4: 1920 -> 960
        self.enc4 = self._make_layer(256, 512, blocks=3)
        self.pool4 = nn.MaxPool1d(2)

        # Layer 5: 960 -> 480
        # 此时长度为 480，完全适合 Transformer 处理
        self.enc5 = self._make_layer(512, 1024, blocks=3)
        self.pool5 = nn.MaxPool1d(2)

        # --- Transformer Bottleneck (核心大脑) ---
        # 算力充足，我们上 8层 Transformer，Hidden=1024
        self.bottleneck_transformer = TransformerBottleneck(
            d_model=1024, nhead=8, num_layers=8, dim_feedforward=4096
        )

        # --- CNN Decoder (负责上采样和细节恢复) ---
        self.up5 = nn.ConvTranspose1d(1024, 512, kernel_size=2, stride=2)
        self.dec5 = self._make_layer(1024 + 512, 512, blocks=2)  # Concat后通道增加

        self.up4 = nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2)
        self.dec4 = self._make_layer(512 + 256, 256, blocks=2)

        self.up3 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self._make_layer(256 + 128, 128, blocks=2)

        self.up2 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self._make_layer(128 + 64, 64, blocks=2)

        self.up1 = nn.ConvTranspose1d(64, 64, kernel_size=2, stride=2)
        self.dec1 = self._make_layer(64 + 64, 32, blocks=1)

        self.out_conv = nn.Conv1d(32, out_channels, kernel_size=1)

    def _make_layer(self, in_c, out_c, blocks):
        ds = None
        if in_c != out_c:
            ds = nn.Sequential(nn.Conv1d(in_c, out_c, 1, bias=False), nn.BatchNorm1d(out_c))
        layers = [ResBlock(in_c, out_c, downsample=ds)]
        for _ in range(1, blocks):
            layers.append(ResBlock(out_c, out_c))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder
        x = self.stem(x)
        e1 = self.enc1(x)  # 15360
        e2 = self.enc2(self.pool1(e1))  # 7680
        e3 = self.enc3(self.pool2(e2))  # 3840
        e4 = self.enc4(self.pool3(e3))  # 1920
        e5 = self.enc5(self.pool4(e4))  # 960

        # Bottleneck
        b = self.pool5(e5)  # 480
        b = self.bottleneck_transformer(b)  # Transformer 处理全局关联

        # Decoder (Skip Connections)
        d5 = self.up5(b)
        # padding handle for odd dimensions just in case, though 15360 is fine
        if d5.size(2) != e5.size(2): d5 = F.interpolate(d5, size=e5.size(2))
        d5 = self.dec5(torch.cat([e5, d5], dim=1))

        d4 = self.up4(d5)
        if d4.size(2) != e4.size(2): d4 = F.interpolate(d4, size=e4.size(2))
        d4 = self.dec4(torch.cat([e4, d4], dim=1))

        d3 = self.up3(d4)
        if d3.size(2) != e3.size(2): d3 = F.interpolate(d3, size=e3.size(2))
        d3 = self.dec3(torch.cat([e3, d3], dim=1))

        d2 = self.up2(d3)
        if d2.size(2) != e2.size(2): d2 = F.interpolate(d2, size=e2.size(2))
        d2 = self.dec2(torch.cat([e2, d2], dim=1))

        d1 = self.up1(d2)
        if d1.size(2) != e1.size(2): d1 = F.interpolate(d1, size=e1.size(2))
        d1 = self.dec1(torch.cat([e1, d1], dim=1))

        out=self.out_conv(d1)
        if self.out_conv.out_channels == 2:
            # Channel 0: ECG Reconstruction (去噪后的心电)
            pred_ecg = out[:, 0:1, :]
            # Channel 1: Label Prediction (预测的 Mask)
            pred_label = out[:, 1:2, :]
            return pred_ecg, pred_label
        else:
            # 单通道输出 (Model B / Model C)
            return out


# ==========================================
# 4. 集成系统 (Ensemble System)
# ==========================================

class ECGEnsembleSystem(nn.Module):
    def __init__(self, device='cpu'):
        super(ECGEnsembleSystem, self).__init__()
        self.device = torch.device(device)

        # 输入维度说明: 3 = Raw ECG + Initial Guess + Quality Map

        # Model A: 高性能生成/去噪模型
        # 输出 2 通道: [0] Clean ECG, [1] Mask
        self.model_a = ECG_TransUNet(in_channels=3, out_channels=2).to(self.device)

        # Model B: 精修识别模型 (输入 A 的结果)
        # 输入 A 的 Clean ECG (1) + A 的 Mask (1) + Quality (1) = 3
        self.model_b = ECG_TransUNet(in_channels=3, out_channels=1).to(self.device)

        # Model C: 鲁棒直连模型
        self.model_c = ECG_TransUNet(in_channels=3, out_channels=1).to(self.device)

    def forward(self, noisy_ecg, initial_guess, quality_map):
        """
        noisy_ecg: (B, 1, L)
        initial_guess: (B, 1, L)
        quality_map: (B, 1, L)
        """
        # 1. 构造 Model A 和 Model C 的输入
        input_raw = torch.cat([noisy_ecg, initial_guess, quality_map], dim=1)

        # --- Run Model A ---
        out_a = self.model_a(input_raw)
        clean_ecg_pred = out_a[:, 0:1, :]
        mask_pred_a = out_a[:, 1:2, :]

        # --- Run Model B ---
        # 使用 A 生成的干净波形作为 B 的输入
        # 注意: 即使 A 生成的不完美，B 也能通过 Transformer 的全局视野进行修正
        input_b = torch.cat([clean_ecg_pred, mask_pred_a, quality_map], dim=1)
        out_b = self.model_b(input_b)

        # --- Run Model C ---
        out_c = self.model_c(input_raw)

        return {
            "clean_ecg": clean_ecg_pred,
            "pred_a": mask_pred_a,
            "pred_b": out_b,
            "pred_c": out_c
        }


# ==========================================
# 5. SQA Model (保持不变，或者也上 TransNet)
# ==========================================
# 这里为了防止过拟合简单回归任务，ResNet-34 足够了，不需要改 Transformer
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
        self.fc = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 1))

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
# 6. Registry (更新)
# ==========================================

class ECGModelRegistry:
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.system = None  # 这里叫 system 更合适，因为是 3个模型
        self.sqa_model = None
        self._init_models()

    def _init_models(self):
        # 实例化集成系统 (包含 TransUNet A, B, C)
        self.system = ECGEnsembleSystem(device=self.device)
        self.sqa_model = ECG_ResNet34_SQA(in_channels=1).to(self.device)

    def reset_models(self):
        self._init_models()
        print("[Registry] TransUNet Ensemble System reset.")

    def get_optimizers(self, lr_seg=1e-3, lr_sqa=1e-3):
        return (
            torch.optim.AdamW(self.system.parameters(), lr=lr_seg, weight_decay=1e-4),  # AdamW 更好
            torch.optim.AdamW(self.sqa_model.parameters(), lr=lr_sqa)
        )

    def export_state_bundle(self):
        return {
            'system_state': self.system.state_dict(),
            'sqa_model_state': self.sqa_model.state_dict(),
            'version': 'TransUNet_Ensemble_v1'
        }

    def load_state_bundle(self, bundle):
        try:
            self.system.load_state_dict(bundle['system_state'])
            self.sqa_model.load_state_dict(bundle['sqa_model_state'])
            return True
        except Exception as e:
            print(f"[Registry] Load failed: {e}")
            return False

    def to_eval_mode(self):
        self.system.eval()
        self.sqa_model.eval()

    def to_train_mode(self):
        self.system.train()
        self.sqa_model.train()