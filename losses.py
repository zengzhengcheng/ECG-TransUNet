import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # inputs: 模型输出的 logits (未经过 sigmoid)
        # targets: 真实标签 0/1
        inputs = torch.sigmoid(logits)

        # 展平以便计算
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)

        return 1 - dice
class SegmentationLoss(nn.Module):
    def __init__(self):
        super(SegmentationLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, pred, target):
        # 0.6 BCE (像素精度) + 0.4 Dice (整体形状)
        return 0.6 * self.bce(pred, target) + 0.4 * self.dice(pred, target)
class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        # 计算一阶差分 (x[t+1] - x[t])
        # pred shape: (B, 1, L)
        pred_grad = torch.abs(pred[..., 1:] - pred[..., :-1])
        target_grad = torch.abs(target[..., 1:] - target[..., :-1])
        return self.l1(pred_grad, target_grad)

class ReconSegLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(ReconSegLoss, self).__init__()
        self.mse = nn.MSELoss()  # 用于去噪重构
        self.bce = nn.BCEWithLogitsLoss()  # 用于分割
        self.dice = DiceLoss()  # 用于分割
        self.alpha = alpha  # 重构损失的权重

    def forward(self, preds, targets):
        """
        preds: (pred_clean_ecg, pred_mask)
        targets: (target_clean_ecg, target_mask)
        """
        pred_ecg, pred_mask = preds
        target_ecg, target_mask = targets

        # Loss 1: 重构 (输入加了噪，我们要让模型输出逼近未加噪的 target_ecg)
        loss_recon = self.mse(pred_ecg, target_ecg)

        # Loss 2: 分割
        loss_seg = 0.6 * self.bce(pred_mask, target_mask) + 0.4 * self.dice(pred_mask, target_mask)

        # 总损失
        return self.alpha * loss_recon + (1 - self.alpha) * loss_seg

class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self, fft_sizes=[1024, 2048, 512], hop_sizes=[120, 240, 50], win_lengths=[600, 1200, 240]):
        super(MultiResolutionSTFTLoss, self).__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        # 维度修正
        if pred.dim() == 3: pred = pred.squeeze(1)
        if target.dim() == 3: target = target.squeeze(1)

        loss = 0.0

        for n_fft, hop_length, win_length in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            window = torch.hann_window(win_length, device=pred.device)

            # 1. 计算 STFT
            pred_stft = torch.stft(pred, n_fft, hop_length, win_length, window, return_complex=True)
            target_stft = torch.stft(target, n_fft, hop_length, win_length, window, return_complex=True)

            # 2. 取幅值
            pred_mag = torch.abs(pred_stft)
            target_mag = torch.abs(target_stft)

            # ======================================================
            # 【核心修复】: 移除不稳定的 Spectral Convergence Loss
            # ======================================================
            # 原来的代码:
            # sc_loss = torch.norm(target_mag - pred_mag, p="fro") / (torch.norm(target_mag, p="fro") + 1e-6)
            # 当 target 接近静音时，这会导致除以 0 爆炸。

            # 方案 A: 只要 Log Magnitude Loss (推荐，稳健)
            mag_loss = self.l1(torch.log(pred_mag + 1e-6), torch.log(target_mag + 1e-6))

            # 方案 B: 如果你非要用 SC Loss，必须防止分母过小
            # sc_loss = torch.norm(target_mag - pred_mag, p="fro") / (torch.norm(target_mag, p="fro") + 1.0) # 加大分母

            loss += mag_loss  # 这里只用 mag_loss

        return loss / len(self.fft_sizes)

class BinaryFocalLoss(nn.Module):
    """
    Focal Loss for Binary Segmentation
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha  # 正样本(R波)的权重系数，设大一点(如0.75)对抗正负样本不平衡
        self.gamma = gamma  # 聚焦系数，越大越关注难样本(建议 2.0)
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: (B, 1, L) 未经过 sigmoid
        # targets: (B, 1, L) 0或1

        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)  # pt 是模型对正确类别的预测概率

        # 构建 alpha 因子
        # 如果 target=1, alpha_t = alpha
        # 如果 target=0, alpha_t = 1 - alpha
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
# ==========================================
# 4. 最终统合: 高级生成损失 (AdvancedGenerativeLoss)
# ==========================================
class PeakConstancyLoss(nn.Module):
    """
    自动寻峰的峰值一致性损失。
    输入宽标签 -> 内部自动提取中心点 -> 强迫中心点预测值为 1.0
    """

    def __init__(self):
        super(PeakConstancyLoss, self).__init__()

    def _get_center_mask(self, masks):
        """
        内部辅助函数：将宽标签 (Block) 还原为单点脉冲 (Impulse)
        """
        # 1. 二值化并Padding (B, 1, L)
        m = (masks > 0.5).float()
        padded = F.pad(m, (1, 1), mode='constant', value=0)

        # 2. 边缘检测
        diff = padded[:, :, 1:] - padded[:, :, :-1]

        # 3. 找到起点和终点
        starts = (diff == 1).nonzero()
        ends = (diff == -1).nonzero()

        # 安全对其
        min_len = min(len(starts), len(ends))
        if min_len == 0:
            return torch.zeros_like(masks)

        starts = starts[:min_len]
        ends = ends[:min_len]

        # 4. 计算中心索引
        # starts[:, 2] 是时间轴起始索引
        # ends[:, 2] 是时间轴结束索引
        center_indices = (starts[:, 2] + ends[:, 2] - 1) // 2
        batch_indices = starts[:, 0]
        channel_indices = starts[:, 1]

        # 5. 生成稀疏掩码
        center_mask = torch.zeros_like(masks)
        center_mask[batch_indices, channel_indices, center_indices] = 1.0

        return center_mask

    def forward(self, pred_ecg, target_mask):
        """
        :param pred_ecg: (B, 1, L) 模型生成的波形
        :param target_mask: (B, 1, L) 原始标签（可以是宽的，也可以是窄的）
        """
        # 1. 动态生成中心点 Mask
        # 这个过程不需要梯度，只是为了找位置
        with torch.no_grad():
            center_mask = self._get_center_mask(target_mask)

        # 2. 计算 Loss
        # 我们只关心 center_mask == 1 的那些点，要求它们的 pred_ecg 接近 1.0

        # 统计有多少个 R 峰 (防止除以 0)
        num_peaks = center_mask.sum()

        if num_peaks == 0:
            # 如果这批数据里一个 R 峰都没有，Loss 为 0，但要保留梯度图连接
            return pred_ecg.sum() * 0.0

        # 计算 L1 距离: |pred - 1.0|
        # 只在中心点位置计算
        diff = torch.abs(pred_ecg - 1.0) * center_mask

        # 求平均 Loss
        loss = diff.sum() / num_peaks

        return loss
class AdvancedGenerativeLoss(nn.Module):
    def __init__(self, w_l1=10.0, w_grad=5.0, w_stft=1.0, w_seg=2.0,w_peak=5.0):
        super(AdvancedGenerativeLoss, self).__init__()

        # 权重超参数
        self.w_l1 = w_l1
        self.w_grad = w_grad
        self.w_stft = w_stft
        self.w_seg = w_seg
        self.w_peak = w_peak

        # 子损失函数
        self.l1_loss = nn.L1Loss()
        self.grad_loss = GradientLoss()
        self.stft_loss = MultiResolutionSTFTLoss()
        self.focal = BinaryFocalLoss(alpha=0.75, gamma=2.0)
        self.peak_loss=PeakConstancyLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, preds, targets,recon_weight=1.0):
        """
        :param preds: Tuple (pred_clean_ecg, pred_mask_logits)
        :param targets: Tuple (target_clean_ecg, target_mask, weight_map)
        """
        pred_ecg, pred_mask_logits = preds
        target_ecg, target_mask, weight_map = targets

        # --- Part A: 信号重构 (Reconstruction) ---
        # 1. L1 Loss (基础形状)
        loss_l1 = self.l1_loss(pred_ecg, target_ecg)

        # 2. Gradient Loss (锐利度/边缘)
        loss_grad = self.grad_loss(pred_ecg, target_ecg)

        # 3. STFT Loss (频域/去噪)
        loss_peak = self.peak_loss(pred_ecg, target_mask)
        loss_stft = self.stft_loss(pred_ecg, target_ecg)
        loss_recon_total = (self.w_l1 * loss_l1) + \
                           (self.w_grad * loss_grad) + \
                           (self.w_stft * loss_stft)+(self.w_peak * loss_peak)
        loss_recon_weighted = loss_recon_total * recon_weight

        # --- Part B: 分割 (Segmentation) ---
        # 4. BCE + Dice
        # loss_bce = (self.bce(pred_mask_logits, target_mask) * weight_map).mean()
        raw_focal = self.focal(pred_mask_logits, target_mask)
        loss_seg_focal = (raw_focal * weight_map).mean()
        loss_dice = self.dice(pred_mask_logits, target_mask)
        # loss_seg = 0.5 * loss_bce + 0.5 * loss_dice
        loss_seg = 0.5 * loss_dice + 0.5 * loss_seg_focal


        # --- 总 Loss 聚合 ---
        total_loss = loss_recon_weighted+self.w_seg * loss_seg

        return total_loss, {
            "l1": loss_l1.item(),
            "grad": loss_grad.item(),
            "stft": loss_stft.item(),
            "seg": loss_seg.item(),
            "peak": loss_peak.item()
        }


class AdvancedSegmentationLoss(nn.Module):
    def __init__(self, focal_alpha=0.75, focal_gamma=2.0):
        super(AdvancedSegmentationLoss, self).__init__()
        # Focal Loss: 关注难分样本 (如低幅度 R 波)
        self.focal = BinaryFocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        # Dice Loss: 关注整体重叠度
        self.dice = DiceLoss()
        # (可选) Boundary Loss 这里暂略，Focal+Dice 通常足够强

    def forward(self, logits, targets, weight_map):
        # 1. Focal Loss (带 Hard Mining 权重)
        loss_focal = (self.focal(logits, targets) * weight_map).mean()

        # 2. Dice Loss
        loss_dice = self.dice(logits, targets)

        # 3. 组合
        # 建议 Dice 权重稍大，因为它直接优化 F1-Score
        return 0.5 * loss_focal + 0.5 * loss_dice


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha  # 惩罚误报 (FP) 的权重
        self.beta = beta  # 惩罚漏报 (FN) 的权重 -> 设大一点！
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits: (B, 1, L) 未经过 sigmoid
        probs = torch.sigmoid(logits)

        # 展平
        probs = probs.view(-1)
        targets = targets.view(-1)

        # 计算 TP, FP, FN
        TP = (probs * targets).sum()
        FP = ((1 - targets) * probs).sum()
        FN = (targets * (1 - probs)).sum()

        # Tversky 系数
        Tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)

        return 1 - Tversky


# --- 更新后的分割专用 Loss ---

class AggressiveSegmentationLoss(nn.Module):
    def __init__(self):
        super(AggressiveSegmentationLoss, self).__init__()

        # 1. Focal Loss: 降低简单背景的权重，挖掘难样本
        # alpha=0.8: 极度偏向正样本(R波)
        self.focal = BinaryFocalLoss(alpha=0.8, gamma=2.0)

        # 2. Tversky Loss: 强制召回
        # beta=0.7: 漏报惩罚极重
        self.tversky = TverskyLoss(alpha=0.3, beta=0.7)

    def forward(self, logits, targets, weight_map=None):
        # Focal Loss 处理像素级分类，利用 weight_map 进行 Hard Negative Mining
        loss_focal = self.focal(logits, targets)
        if weight_map is not None:
            loss_focal = (loss_focal * weight_map).mean()
        else:
            loss_focal = loss_focal.mean()

        # Tversky Loss 关注整体形状和召回率
        loss_tversky = self.tversky(logits, targets)

        # 组合：Tversky 主导，Focal 辅助
        return 0.4 * loss_focal + 0.6 * loss_tversky