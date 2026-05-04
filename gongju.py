import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pywt
from scipy import signal
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from scipy.signal import find_peaks
def select_final_labels(all_preds_dict):
    final_preds = []  # 存储最终的预测标签
    for idx, pred_probs_list in all_preds_dict.items():
        # 获取每个预测数组的标签
        pred_labels = [np.argmax(prob) for prob in pred_probs_list]  # 每个预测数组选择概率最大的位置作为标签

        if len(pred_probs_list) == 1:
            # 只有一个预测数组，直接选择概率最大的标签
            final_pred = pred_labels[0]
        else:
            # 有多个预测数组，首先统计标签出现次数
            unique_labels, counts = np.unique(pred_labels, return_counts=True)
            max_count = np.max(counts)

            if max_count >= 2:
                # 如果某个标签出现两次或更多次，选择该标签
                final_pred = unique_labels[np.argmax(counts)]
            else:
                # 如果标签都不相同，选择概率最大的标签
                max_prob_idx = np.argmax([np.max(prob) for prob in pred_probs_list])  # 选择概率最大的预测
                final_pred = pred_labels[max_prob_idx]

        final_preds.append(final_pred)

    return np.array(final_preds)
def visualize_and_save(data, labels, pred_r_peak, epoch,step, img_dir,accuracy):
    print("start visualization")

    # print(data.shape,labels.shape)
    # print(len(pred_r_peak))
    # print(pred_r_peak.shape)
    # 选择最后30秒的数据进行可视化
    start_idx = 0  # 选择最后30秒的数据
    data_segment = data[0, 0,start_idx:start_idx + 153600]
    data_segment=data_segment*100
    if (len(labels.shape) == 2):
        labels_segment = labels[0, start_idx:start_idx + 153600]
    else:
        labels_segment = labels[0, 0, start_idx:start_idx + 153600]
    pred_segment = pred_r_peak[ start_idx:start_idx + 153600]
    print("输出预测值的和",pred_segment.sum())
    # 创建图表
    plt.figure(figsize=(15, 6))

    # 绘制原始数据
    plt.subplot(3, 1, 1)
    plt.plot(data_segment)
    plt.title(f"ECG Data (Step {step})")

    # 绘制真实标签
    plt.subplot(3, 1, 2)
    plt.plot(labels_segment, label="True Labels")
    plt.legend()
    plt.title("True Labels")

    # 绘制预测标签
    plt.subplot(3, 1, 3)
    plt.plot(pred_segment, label="Predicted Labels", linestyle="--")
    plt.legend()
    plt.title(f"Predicted Labels, Accuracy: {accuracy}")

    # 保存图片
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    plt.tight_layout()
    plt.savefig(f'{img_dir}/_Epoch{epoch}_Step{step}.png')
    plt.close()
def gan_visualize_and_save(data, labels, clean_ecg,pred_r_peak,pred_data, epoch,step, img_dir,accuracy=None):
    print("start visualization")

    # print(data.shape,labels.shape)
    # print(len(pred_r_peak))
    # print(pred_r_peak.shape)
    # 选择最后30秒的数据进行可视化
    start_idx = 0  # 选择最后30秒的数据
    data_segment = data[0, 0,start_idx:start_idx + 153600]
    data_segment=data_segment*100
    if(len(labels.shape)==2):
        labels_segment = labels[0, start_idx:start_idx + 153600]
    else:
        labels_segment = labels[0,0, start_idx:start_idx + 153600]
    pred_segment = pred_r_peak[ start_idx:start_idx + 153600]
    pred_data=pred_data[0, 0,start_idx:start_idx + 153600]
    clean_data=clean_ecg[0, 0,start_idx:start_idx + 153600]
    print("输出预测值的和",pred_segment.sum())
    # 创建图表
    plt.figure(figsize=(15, 6))

    # 绘制原始数据
    plt.subplot(5, 1, 1)
    plt.plot(data_segment)
    plt.title(f"ECG Data (Step {step})")
    # 绘制原始数据
    plt.subplot(5, 1, 2)
    plt.plot(clean_data)
    plt.title(f"ECG Clean Data (Step {step})")
    # 绘制原始数据
    plt.subplot(5, 1, 3)
    plt.plot(pred_data)
    plt.title(f"ECG Pred Data (Step {step})")
    # 绘制真实标签
    plt.subplot(5, 1, 4)
    plt.plot(labels_segment, label="True Labels")
    plt.legend()
    plt.title("True Labels")

    # 绘制预测标签
    plt.subplot(5, 1, 5)
    plt.plot(pred_segment, label="Predicted Labels", linestyle="--")
    plt.legend()
    plt.title(f"Predicted Labels, Accuracy: {accuracy}")

    # 保存图片
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    plt.tight_layout()
    plt.savefig(f'{img_dir}/_Epoch{epoch}_Step{step}.png')
    plt.close()


def get_optimizer_and_scheduler(model, session_epochs, lr=1e-3, weight_decay=0.01):
    """
    初始化优化器和调度器。

    :param session_epochs: 本次要训练多少轮 (不是累计轮数，而是新增轮数)
    :param lr: 基础学习率
    """
    # 1. 优化器 (AdamW)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 2. 调度器策略
    # 如果训练轮数太少（比如只微调1轮），直接用 Cosine
    if session_epochs <= 1:
        scheduler = CosineAnnealingLR(optimizer, T_max=1, eta_min=1e-6)
        return optimizer, scheduler

    # A. 预热 (Warmup) - 占本次训练的 10% 或至少 1 轮
    warmup_epochs = max(1, int(session_epochs * 0.1))

    scheduler_warmup = LinearLR(
        optimizer,
        start_factor=0.01,  # 从 lr*0.01 开始
        end_factor=1.0,  # 爬升到 lr
        total_iters=warmup_epochs
    )

    # B. 余弦退火 (Cosine Decay) - 剩下的轮数
    main_epochs = session_epochs - warmup_epochs
    scheduler_cosine = CosineAnnealingLR(
        optimizer,
        T_max=main_epochs,
        eta_min=1e-6  # 最小衰减到 1e-6
    )

    # C. 串联
    scheduler = SequentialLR(
        optimizer,
        schedulers=[scheduler_warmup, scheduler_cosine],
        milestones=[warmup_epochs]
    )

    return optimizer, scheduler
def loss_function(pred_output, true_labels, lambda_l1=0.01): # lambda_l1 是 L1 正则化系数，可调整
    criterion = nn.CrossEntropyLoss()
    ce_loss = criterion(pred_output, true_labels) # 交叉熵损失
    # 提取预测输出中对应于 R 峰标签 (假设标签 1 和 2 的索引是 1 和 2)
    r_peak_probs = pred_output[:,  [1, 2]] #  假设 pred_output 形状是 [batch_size, sequence_length, num_classes]

    # 计算 L1 正则化项
    l1_regularization = torch.sum(torch.abs(r_peak_probs))

    # 总损失 = 交叉熵损失 + L1 正则化项
    total_loss = ce_loss + lambda_l1 * l1_regularization
    return total_loss


def generate_weight_from_labels(labels, is_soft_label=False, to_tensor=True):
    """
    根据【已经扩展过】的标签生成 Target 和 WeightMap。

    :param labels: numpy array or tensor (Batch, Length) 或 (Length,)
                   - 硬标签模式: 0(背景), 1(心跳), 2(噪声)
                   - 软标签模式: 0.0~1.0(心跳概率), >=2.0(噪声标记)
    :param is_soft_label: bool, 是否为软标签。
                          True: Target 会保留 0.x 的数值；
                          False: Target 只会是 0 或 1。
    :param to_tensor: bool, 是否转换为符合 BCE Loss 的 PyTorch Tensor
    :return: (target, weight_map)
    """
    # 1. 统一转为 Numpy
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    # 2. 初始化
    # Target 默认为 0.0 (背景)
    target = np.zeros_like(labels, dtype=np.float32)
    # Weight 默认为 1.0 (普通背景权重)
    weight_map = np.ones_like(labels, dtype=np.float32)

    # =========================================================
    # 3. 分情况处理
    # =========================================================
    if is_soft_label:
        # --- 软标签模式 ---
        # 假设：心跳是 (0, 1] 之间的小数，噪声标记为 >= 2.0 的数

        # A. 心跳区域 (大于0 且 小于等于1)
        # 即使标签是 0.001 这种极小值，我们也认为是心跳区域的一部分，或者是边缘
        # 这里加个极小阈值 1e-3 防止浮点误差
        mask_beat = (labels > 1e-3) & (labels <= 1.0)

        # 【关键】Target 直接等于软标签的值 (保留高斯分布形态)
        target[mask_beat] = labels[mask_beat]
        weight_map[mask_beat] = 10.0

        # B. 噪声区域 (>= 2.0)
        mask_error = (labels >= 1.9)  # 稍微放宽一点点防浮点误差

    else:
        # --- 硬标签模式 ---
        # 只有整数 0, 1, 2

        # A. 心跳区域
        mask_beat = (labels == 1)
        target[mask_beat] = 1.0
        weight_map[mask_beat] = 10.0

        # B. 噪声区域
        mask_error = (labels == 2)

    # =========================================================
    # 4. 噪声/错误区域处理 (两种模式通用)
    # =========================================================
    # 逻辑：强制 Target=0, Weight=50
    # 放在最后执行，确保如果标签有重叠(比如软标签边缘碰到了噪声)，噪声处理覆盖心跳处理
    if np.any(mask_error):
        target[mask_error] = 0.0
        weight_map[mask_error] = 50.0

    # =========================================================
    # 5. 格式转换
    # =========================================================
    if to_tensor:
        target_t = torch.from_numpy(target).float()
        weight_t = torch.from_numpy(weight_map).float()

        # 维度修正：保证输出是 (Batch, 1, Length) 或者是 (1, 1, Length)
        # 以适应 BCEWithLogitsLoss 的输入要求 (N, C, ...)

        if target_t.dim() == 1:
            # (L,) -> (1, L)
            target_t = target_t.unsqueeze(0)
            weight_t = weight_t.unsqueeze(0)

        if target_t.dim() == 2:
            # (B, L) -> (B, 1, L)
            target_t = target_t.unsqueeze(1)
            weight_t = weight_t.unsqueeze(1)

        return target_t, weight_t
    else:
        return target, weight_map
# 计算损失率（准确率）
def calculate_accuracy(final_preds, true_labels):
    print(f"Type of final_preds: {type(final_preds)}")
    print(f"Type of true_labels: {type(true_labels)}")

    # 方案一:  将 PyTorch Tensor 转换为 NumPy 数组再用 np.sum() (如果确保在 CPU 上)
    correct = np.sum((final_preds == true_labels).cpu().numpy())  # 如果 Tensor 在 GPU 上需要先 .cpu()

    #correct = np.sum(final_preds == true_labels)
    total = len(true_labels)
    accuracy = correct / total
    return accuracy
def wavelet_filter_ecg(ecg_signal, wavelet_name='db4', wavelet_level=4, threshold_method='硬阈值', threshold_value=None):
    """
    对心电信号进行小波滤波降噪 (与之前代码相同)
    """
    wavelet = pywt.Wavelet(wavelet_name)
    coeffs = pywt.wavedec(ecg_signal, wavelet, level=wavelet_level)
    coeffs_thresholded = list(coeffs)
    for i in range(1, len(coeffs_thresholded)):
        detail_coeffs = coeffs_thresholded[i]
        if threshold_value is None:
            threshold = np.median(np.abs(detail_coeffs)) * 0.6745
        else:
            threshold = threshold_value
        if threshold_method == '硬阈值':
            coeffs_thresholded[i] = pywt.threshold(detail_coeffs, threshold, mode='hard')
        elif threshold_method == '软阈值':
            coeffs_thresholded[i] = pywt.threshold(detail_coeffs, threshold, mode='soft')
        else:
            raise ValueError("Threshold method must be '硬阈值' or '软阈值'")
    ecg_signal_filtered = pywt.waverec(coeffs_thresholded, wavelet)
    return ecg_signal_filtered
def xiaobo(ecg_signal,sample_rate=512,duration=15):
    time = np.arange(0, duration, 1 / sample_rate)
    # 小波滤波
    wavelet_filtered_signal = wavelet_filter_ecg(ecg_signal, wavelet_name='db4', wavelet_level=4,
                                                 threshold_method='硬阈值')
    return wavelet_filtered_signal
def butterBandPassFilter(lowcut, highcut, samplerate, order):
    "生成巴特沃斯带通滤波器"
    semiSampleRate = samplerate*0.5
    low = lowcut / semiSampleRate
    high = highcut / semiSampleRate
    b,a = signal.butter(order,[low,high],btype='bandpass')

    return b,a

def butterBandStopFilter(lowcut, highcut, samplerate, order):
    "生成巴特沃斯带阻滤波器"
    semiSampleRate = samplerate*0.5
    low = lowcut / semiSampleRate
    high = highcut / semiSampleRate
    b,a = signal.butter(order,[low,high],btype='bandstop')

    return b,a
def lvbo(x, iSampleRate):
    # 1. 带通滤波：使用 filtfilt 替代 lfilter
    b, a = butterBandPassFilter(3, 70, iSampleRate, order=4)
    # x = signal.lfilter(b, a, x)  <-- 删除这一行 (会有延迟)
    x = signal.filtfilt(b, a, x)   # <-- 换成这一行 (零相位，无延迟)

    # 2. 带阻滤波 (50Hz工频干扰)：同样使用 filtfilt
    b, a = butterBandStopFilter(48, 52, iSampleRate, order=2)
    # x = signal.lfilter(b, a, x)  <-- 删除这一行
    x = signal.filtfilt(b, a, x)   # <-- 换成这一行

    # 小波滤波通常相位偏移很小，可以保留
    # 如果 wavelet_filter_ecg 内部用了 shift 相关的操作要注意，但通常 db4 是没问题的
    wavelet_filtered_signal = wavelet_filter_ecg(x, wavelet_name='db4', wavelet_level=4,
                                                 threshold_method='硬阈值')
    return wavelet_filtered_signal


class ECGSynthesizer(nn.Module):
    def __init__(self, sampling_rate=512, device='cpu'):
        super(ECGSynthesizer, self).__init__()
        self.device = device
        self.fs = sampling_rate

        # 1. 修改时间窗口: 增加到 1.0秒，保证足够容纳 P和T，且留出边缘缓冲
        self.duration = 1.0
        self.kernel_size = int(self.duration * self.fs)

        # 强制奇数长度，保证有绝对中心点
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1

        self.padding = self.kernel_size // 2

        # 2. 【核心修复】: 时间轴必须关于 0 对称！
        # 范围从 -0.5 到 +0.5，这样 t=0 就在数组的正中间
        half_duration = self.duration / 2
        t = torch.linspace(-half_duration, half_duration, self.kernel_size)

        # 参数调整 (位置参数 mu 保持不变，因为现在坐标系是对称的了)
        params = {
            'P': (0.15, -0.20, 0.02),  # P波在 R 前 0.2s
            'Q': (-0.15, -0.02, 0.005),
            'R': (1.00, 0.00, 0.008),  # R波在 0 (正中心)
            'S': (-0.25, 0.02, 0.008),
            'T': (0.30, 0.30, 0.06)  # T波在 R 后 0.3s
        }

        template = torch.zeros_like(t)
        for wave, (A, mu, sigma) in params.items():
            gauss = A * torch.exp(- (t - mu) ** 2 / (2 * sigma ** 2))
            template += gauss

        template = template / template.max()
        self.kernel = template.view(1, 1, -1).to(device)

    # _skeletonize 和 forward 函数不需要改动，保持原样即可
    def _skeletonize(self, labels):
        mask = (labels > 0.5).float()
        padded = F.pad(mask, (1, 1), mode='constant', value=0)
        diff = padded[:, :, 1:] - padded[:, :, :-1]

        starts = (diff == 1).nonzero()
        ends = (diff == -1).nonzero()

        min_len = min(len(starts), len(ends))
        if min_len == 0: return torch.zeros_like(labels)

        starts = starts[:min_len]
        ends = ends[:min_len]

        center_indices = (starts[:, 2] + ends[:, 2] - 1) // 2
        batch_indices = starts[:, 0]

        new_labels = torch.zeros_like(labels)
        new_labels[batch_indices, 0, center_indices] = 1.0
        return new_labels

    def forward(self, labels):
        x = labels.float()
        seq_len = x.shape[-1]
        x = x.view(-1, 1, seq_len)

        x_impulse = self._skeletonize(x)
        ideal_ecg = F.conv1d(x_impulse, self.kernel, padding=self.padding)
        ideal_ecg = torch.clamp(ideal_ecg, -1.0, 1.0)

        return ideal_ecg


class AdaptiveECGSynthesizer(nn.Module):
    def __init__(self, sampling_rate=512, device='cpu'):
        super(AdaptiveECGSynthesizer, self).__init__()
        self.device = device
        self.fs = sampling_rate

        # 预先计算时间轴的基础网格 (为了速度，不用每次都 linspace)
        # 我们假设最慢心率 40bpm -> 1.5秒，这足够覆盖所有情况
        self.max_kernel_len = int(1.5 * self.fs)
        self.base_t = torch.linspace(-0.75, 0.75, self.max_kernel_len).to(device)

    def _get_dynamic_kernel(self, current_bpm):
        """
        根据当前 BPM 动态生成卷积核
        """
        # 1. 确定缩放系数
        # 以 60 bpm 为基准。BPM 越高，scale 越小，波形越窄
        # 限制 scale 最大为 1.0 (即使心率很慢，也不要让波形变得无限宽)
        scale = min(1.0, 60.0 / current_bpm)

        # 2. 确定窗口长度
        # 理论周期 = 60 / bpm
        # 窗口长度取周期的 80% 即可覆盖 P-QRS-T，留出 20% 的平直基线防止重叠
        window_sec = (60.0 / current_bpm) * 0.8

        # 限制窗口最小 0.4秒 (对应 150bpm)，最大 1.0秒
        window_sec = max(0.4, min(1.0, window_sec))

        kernel_size = int(window_sec * self.fs)
        if kernel_size % 2 == 0: kernel_size += 1

        # 3. 从预计算的 base_t 中截取一段对称的时间轴
        # 这样比每次都 linspace 要快微乎其微，但更优雅
        half_len = kernel_size // 2
        center_idx = self.max_kernel_len // 2
        # 取出 [-half, +half] 的时间段
        t_slice = self.base_t[center_idx - half_len: center_idx + half_len + 1]

        # 4. 生成波形 (高斯叠加)
        template = torch.zeros_like(t_slice)

        # 参数: (幅度A, 位置mu, 宽度sigma)
        # 关键：位置和宽度都乘以 scale
        params = {
            'P': (0.15, -0.20 * scale, 0.02 * scale),
            'Q': (-0.15, -0.02 * scale, 0.005 * scale),
            'R': (1.00, 0.00, 0.008 * scale),
            'S': (-0.25, 0.02 * scale, 0.008 * scale),
            'T': (0.30, 0.30 * scale, 0.06 * scale)
        }

        for _, (A, mu, sigma) in params.items():
            gauss = A * torch.exp(- (t_slice - mu) ** 2 / (2 * sigma ** 2))
            template += gauss

        template = template / (template.max() + 1e-8)  # 归一化

        # 返回形状 (Out=1, In=1, K)
        return template.view(1, 1, -1)

    def _skeletonize(self, labels):
        """将宽标签块还原为单点脉冲"""
        mask = (labels > 0.5).float()
        # 前后 pad 0 以处理边缘
        padded = F.pad(mask, (1, 1), mode='constant', value=0)
        diff = padded[:, :, 1:] - padded[:, :, :-1]

        starts = (diff == 1).nonzero()
        ends = (diff == -1).nonzero()

        min_len = min(len(starts), len(ends))
        if min_len == 0: return torch.zeros_like(labels)

        starts = starts[:min_len]
        ends = ends[:min_len]

        # 计算中心
        center_indices = (starts[:, 2] + ends[:, 2] - 1) // 2
        batch_indices = starts[:, 0]

        new_labels = torch.zeros_like(labels)
        new_labels[batch_indices, 0, center_indices] = 1.0
        return new_labels

    def forward(self, labels):
        x = labels.float()
        # 强转维度 (Total_Batch, 1, Len)
        x = x.view(-1, 1, x.shape[-1])

        # 1. 骨架化
        x_impulse = self._skeletonize(x)

        # 2. 【自适应计算 BPM】
        # 统计所有样本中的 R 峰总数
        total_peaks = x_impulse.sum().item()

        # 计算总时长 (秒) = Batch大小 * 单条时长
        total_seconds = x.shape[0] * (x.shape[-1] / self.fs)

        # 计算平均 BPM
        if total_seconds > 0 and total_peaks > 0:
            avg_bpm = (total_peaks / total_seconds) * 60.0
        else:
            avg_bpm = 80.0  # 默认值，防止空数据

        # 【策略】: 为了防止局部密集导致重叠，我们取一个 "保守 BPM"
        # 比如：取 calculated_bpm 和 90 的较大值
        # 这样即使算出来只有 60，我们也按 90 生成，波形窄一点没坏处
        # 如果算出来是 150，我们就按 150 * 1.1 = 165 生成，更保险
        target_bpm = max(90.0, avg_bpm * 1.1)

        # 3. 动态生成核
        # 注意：这个核不参与梯度更新，只是用来生成 Target
        kernel = self._get_dynamic_kernel(target_bpm)
        padding = kernel.shape[-1] // 2

        # 4. 卷积
        ideal_ecg = F.conv1d(x_impulse, kernel, padding=padding)

        # 限制范围
        ideal_ecg = torch.clamp(ideal_ecg, -1.0, 1.0)

        return ideal_ecg





# def calc_ecg_accuracy(logits, targets, tolerance=5, threshold=0.5):
#     """
#     计算心电R峰检测的准确率 (Precision, Recall, F1)。
#     【新增功能】: 自动修正输入形状，兼容 (B,1,L), (B,L), (L,) 等各种维度。
#
#     :param logits: 模型输出 (未经过Sigmoid)
#     :param targets: 真实标签 (建议是未扩展的，或者扩展后中心为1的)
#     :param tolerance: 允许误差范围 (点数)
#     :param threshold: 判定阈值
#     """
#
#     # 1. 统一转为 Numpy 数组
#     if isinstance(logits, torch.Tensor):
#         logits = logits.detach().cpu().numpy()
#     if isinstance(targets, torch.Tensor):
#         targets = targets.detach().cpu().numpy()
#
#     # 2. 内部辅助函数：统一维度到 (Batch, Length)
#     def _standardize_shape(arr):
#         # 情况 A: 单条数据 (L,) -> (1, L)
#         if arr.ndim == 1:
#             return arr[np.newaxis, :]
#         # 情况 B: 标准 PyTorch 输出 (B, 1, L) -> (B, L)
#         elif arr.ndim == 3 and arr.shape[1] == 1:
#             return arr.squeeze(axis=1)
#         # 情况 C: 通道在后 (B, L, 1) -> (B, L)
#         elif arr.ndim == 3 and arr.shape[2] == 1:
#             return arr.squeeze(axis=2)
#         # 情况 D: 已经是 (B, L) -> 保持
#         return arr
#
#     # 3. 应用形状修正
#     # 先做 sigmoid 转概率，再修形状
#     probs = 1.0 / (1.0 + np.exp(-logits))  # Sigmoid
#     probs = _standardize_shape(probs)
#     targets = _standardize_shape(targets)
#
#     # 简单校验
#     if probs.shape != targets.shape:
#         print(f"[Metric Warning] Shape mismatch after fix: Pred {probs.shape} vs Target {targets.shape}")
#         # 尝试截断对齐（防止长度有细微差别）
#         min_len = min(probs.shape[1], targets.shape[1])
#         probs = probs[:, :min_len]
#         targets = targets[:, :min_len]
#
#     batch_size = probs.shape[0]
#     total_tp = 0
#     total_fp = 0
#     total_fn = 0
#
#     # 4. 开始计算
#     for b in range(batch_size):
#         # A. 从概率图中找峰值 (Local Maxima > threshold)
#         # find_peaks 自带了寻找局部最大值的功能，比单纯 threshold 更好
#         pred_indices, _ = find_peaks(probs[b], height=threshold, distance=tolerance)
#
#         # B. 从真值找峰值
#         # 兼容扩展过的标签(Target为1)或原始稀疏标签
#         true_indices = np.where(targets[b] >= 0.99)[0]  # 只要接近1都算
#
#         # C. 匹配算法
#         matched_true = set()
#         tp = 0
#
#         for p in pred_indices:
#             hit = False
#             # 在预测点 p 的左右 tolerance 范围内找真值
#             # 优化：只搜索附近的真值，而不是遍历所有
#             # 这里简单遍历，因为R峰数量不多 (30秒大概20-40个)，不会慢
#             for t in true_indices:
#                 if abs(p - t) <= tolerance:
#                     if t not in matched_true:
#                         matched_true.add(t)
#                         tp += 1
#                         hit = True
#                         break  # 这个预测点匹配到了，跳出
#
#             if not hit:
#                 total_fp += 1  # 误报 (False Positive)
#
#         # 漏报 (False Negative) = 总真值 - 匹配到的真值
#         fn = len(true_indices) - tp
#         total_fn += fn
#         total_tp += tp
#
#     # 5. 汇总计算
#     epsilon = 1e-7
#     precision = total_tp / (total_tp + total_fp + epsilon)
#     recall = total_tp / (total_tp + total_fn + epsilon)
#     f1 = 2 * precision * recall / (precision + recall + epsilon)
#
#     return {
#         "acc_precision": precision,
#         "acc_recall": recall,
#         "acc_f1": f1
#     }

def normalize_input_shape(x,device=None):
    """
    自动维度修正工具。
    无论输入是什么奇形怪状，统一修正为标准格式: (Batch_Size, Channels, Length)

    支持的输入情况:
    1. (L,)           -> (1, 1, L)   [单条数据]
    2. (B, L)         -> (B, 1, L)   [批量数据，无通道维]
    3. (L, C)         -> (1, C, L)   [单条数据，通道在后]
    4. (B, L, C)      -> (B, C, L)   [批量数据，通道在后 - Keras/TF风格]
    5. (B, C, L)      -> (B, C, L)   [标准格式，保持不变]

    :param x: torch.Tensor 或 numpy.ndarray
    :return: torch.Tensor (B, C, L)
    """
    # 1. 统一转为 Tensor
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(x).float()

    # 2. 获取维度信息
    dim_count = x.dim()
    shape = x.shape

    # --- 情况 A: 只有 1 维 (Length,) ---
    if dim_count == 1:
        # 这里的 x 是一条单纯的信号
        # 扩展出 Batch 和 Channel
        x = x.unsqueeze(0).unsqueeze(0)  # (L,) -> (1, 1, L)

    # --- 情况 B: 2 维 (Batch, Length) 或 (Length, Channel) ---
    elif dim_count == 2:
        # 我们假设信号长度 (Length) 肯定比 BatchSize 或 Channels 要大得多
        # 比如 15360 vs 16
        if shape[0] > shape[1]:
            # 认为是 (Length, Channel) 或 (Length, 1) -> 转为 (1, C, L)
            # 这种很少见，但为了健壮性加上
            x = x.permute(1, 0).unsqueeze(0)
        else:
            # 认为是 (Batch, Length) -> 转为 (B, 1, L)
            # 这是最常见的错误来源
            x = x.unsqueeze(1)

            # --- 情况 C: 3 维 (Batch, Length, Channel) 或 (Batch, Channel, Length) ---
    elif dim_count == 3:
        # 判断哪个维度是长度。通常 Length 是最大的。
        # 如果最后一个维度最小 (比如 1)，而中间维度很大，说明是 (B, L, C)
        if shape[-1] < shape[-2]:
            # 说明可能是 (Batch, Length, Channel) -> 修正为 (Batch, Channel, Length)
            x = x.permute(0, 2, 1)

        # 否则已经是 (Batch, Channel, Length)，保持不变

    # 3. 放到正确的设备上 (可选)
    if device:
        x = x.to(device)
    return x


def calc_ecg_accuracy(logits, targets, tolerance=20, threshold=0.5):
    """
    计算准确率 (修正版)
    1. tolerance 默认提高到 20 (约40ms @ 512Hz)
    2. find_peaks 增加 distance 参数，防止对同一个R波重复检测
    """
    # ... (维度修正代码保持不变, _standardize_shape 那些) ...
    # (为了篇幅，这里直接从数据处理开始)

    if isinstance(logits, torch.Tensor): logits = logits.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor): targets = targets.detach().cpu().numpy()

    def _standardize_shape(arr):
        # 情况 A: 单条数据 (L,) -> (1, L)
        if arr.ndim == 1:
            return arr[np.newaxis, :]
        # 情况 B: 标准 PyTorch 输出 (B, 1, L) -> (B, L)
        elif arr.ndim == 3 and arr.shape[1] == 1:
            return arr.squeeze(axis=1)
        # 情况 C: 通道在后 (B, L, 1) -> (B, L)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            return arr.squeeze(axis=2)
        # 情况 D: 已经是 (B, L) -> 保持
        return arr

    # 3. 应用形状修正
    # 先做 sigmoid 转概率，再修形状
    probs = 1.0 / (1.0 + np.exp(-logits))  # Sigmoid
    probs = _standardize_shape(probs)
    targets = _standardize_shape(targets)
    if probs.shape != targets.shape:
        print(f"[Metric Warning] Shape mismatch after fix: Pred {probs.shape} vs Target {targets.shape}")
        min_len = min(probs.shape[1], targets.shape[1])
        probs = probs[:, :min_len]
        targets = targets[:, :min_len]

    batch_size = probs.shape[0]
    total_tp, total_fp, total_fn = 0, 0, 0

    # 用于调试：记录平均偏差距离
    dist_errors = []

    for b in range(batch_size):
        # 【关键修改 A】: 设置最小峰间距 distance
        # 正常心跳间隔一般 > 200 点。设置 distance=100 可以防止在一个R波内检测出两个峰
        pred_indices, _ = find_peaks(probs[b], height=threshold, distance=50)

        # 真值寻找
        true_indices = np.where(targets[b] >= 0.5)[0]  # 兼容软标签

        # 为了防止扩展标签导致真值是一坨连续的索引，我们需要对真值也做聚类取中心
        # 简单的做法是：既然我们有 pred_indices, 我们看每个 true cluster 里有没有 pred
        # 但为了通用，我们假设 targets 最好是稀疏的。如果是宽标签，取连续块的中心。
        if len(true_indices) > 0:
            # 简易骨架化真值 (处理宽标签)
            diff = true_indices[1:] - true_indices[:-1]
            split_at = np.where(diff > 5)[0] + 1
            true_clusters = np.split(true_indices, split_at)
            true_peaks = np.array([int(np.mean(c)) for c in true_clusters if len(c) > 0])
        else:
            true_peaks = np.array([])

        # --- 匹配逻辑 ---
        matched_true = set()

        for p in pred_indices:
            hit = False
            best_dist = float('inf')
            best_t = -1

            # 在容差范围内寻找最近的真值
            for t in true_peaks:
                dist = abs(p - t)
                if dist <= tolerance:
                    if dist < best_dist:
                        best_dist = dist
                        best_t = t

            if best_t != -1:
                # 找到了匹配
                if best_t not in matched_true:
                    matched_true.add(best_t)
                    total_tp += 1
                    dist_errors.append(best_dist)
                    hit = True
                else:
                    # 这个真值已经被前面的预测点匹配了
                    # 说明在同一个真值附近预测了两个点 -> 视为误报
                    total_fp += 1

            if not hit:
                total_fp += 1  # 没找到真值

        # 漏报
        total_fn += len(true_peaks) - len(matched_true)

    # 打印一下平均偏差，帮你确认问题
    if len(dist_errors) > 0:
        avg_shift = np.mean(dist_errors)
        # 可以在日志里或者这里 print 出来看看
        # print(f"[DEBUG] Avg Peak Shift: {avg_shift:.2f} samples")

    epsilon = 1e-7
    precision = total_tp / (total_tp + total_fp + epsilon)
    recall = total_tp / (total_tp + total_fn + epsilon)
    f1 = 2 * precision * recall / (precision + recall + epsilon)

    # 简单的 Accuracy = (TP) / (TP + FP + FN)
    # (即 Jaccard Index，非像素级 Accuracy，更符合医学直觉)
    accuracy = total_tp / (total_tp + total_fp + total_fn + epsilon)

    return {
        "acc_accuracy": accuracy,
        "acc_precision": precision,
        "acc_recall": recall,
        "acc_f1": f1
    }

def calc_ecg_accuracy_detailed(logits, targets, tolerance=20, threshold=0.5):
    """
    计算心电R峰检测的详细评估指标
    
    :param logits: 模型输出 (未经过Sigmoid)
    :param targets: 真实标签
    :param tolerance: 允许误差范围 (点数)
    :param threshold: 判定阈值
    :return: 详细评估指标字典
    """
    # 1. 统一转为 Numpy 数组
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # 2. 内部辅助函数：统一维度到 (Batch, Length)
    def _standardize_shape(arr):
        if arr.ndim == 1:
            return arr[np.newaxis, :]
        elif arr.ndim == 3 and arr.shape[1] == 1:
            return arr.squeeze(axis=1)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            return arr.squeeze(axis=2)
        return arr

    # 3. 应用形状修正
    probs = 1.0 / (1.0 + np.exp(-logits))  # Sigmoid
    probs = _standardize_shape(probs)
    targets = _standardize_shape(targets)
    
    if probs.shape != targets.shape:
        print(f"[Metric Warning] Shape mismatch after fix: Pred {probs.shape} vs Target {targets.shape}")
        min_len = min(probs.shape[1], targets.shape[1])
        probs = probs[:, :min_len]
        targets = targets[:, :min_len]

    batch_size = probs.shape[0]
    total_tp, total_fp, total_fn = 0, 0, 0
    dist_errors = []
    per_sample_metrics = []
    false_positives = []
    false_negatives = []

    for b in range(batch_size):
        # 预测峰值检测
        pred_indices, _ = find_peaks(probs[b], height=threshold, distance=50)

        # 真值处理
        true_indices = np.where(targets[b] >= 0.5)[0]
        if len(true_indices) > 0:
            # 简易骨架化真值 (处理宽标签)
            diff = true_indices[1:] - true_indices[:-1]
            split_at = np.where(diff > 5)[0] + 1
            true_clusters = np.split(true_indices, split_at)
            true_peaks = np.array([int(np.mean(c)) for c in true_clusters if len(c) > 0])
        else:
            true_peaks = np.array([])

        # 匹配逻辑
        matched_true = set()
        matched_pairs = []  # 存储匹配的 (预测, 真实) 对
        sample_fp = []  # 当前样本的误报
        sample_fn = []  # 当前样本的漏报

        for p in pred_indices:
            hit = False
            best_dist = float('inf')
            best_t = -1

            # 在容差范围内寻找最近的真值
            for t in true_peaks:
                dist = abs(p - t)
                if dist <= tolerance:
                    if dist < best_dist:
                        best_dist = dist
                        best_t = t

            if best_t != -1:
                if best_t not in matched_true:
                    matched_true.add(best_t)
                    total_tp += 1
                    dist_errors.append(best_dist)
                    matched_pairs.append((p, best_t))
                    hit = True
                else:
                    total_fp += 1
                    sample_fp.append(p)
            if not hit:
                total_fp += 1
                sample_fp.append(p)

        # 漏报
        fn = len(true_peaks) - len(matched_true)
        total_fn += fn
        # 记录漏报的真实R波位置
        for t in true_peaks:
            if t not in matched_true:
                sample_fn.append(t)

        # 计算当前样本的指标
        epsilon = 1e-7
        sample_tp = len(matched_pairs)
        sample_fp_count = len(sample_fp)
        sample_fn_count = len(sample_fn)
        sample_precision = sample_tp / (sample_tp + sample_fp_count + epsilon)
        sample_recall = sample_tp / (sample_tp + sample_fn_count + epsilon)
        sample_f1 = 2 * sample_precision * sample_recall / (sample_precision + sample_recall + epsilon)
        
        per_sample_metrics.append({
            "precision": sample_precision,
            "recall": sample_recall,
            "f1": sample_f1,
            "tp": sample_tp,
            "fp": sample_fp_count,
            "fn": sample_fn_count
        })
        
        false_positives.extend(sample_fp)
        false_negatives.extend(sample_fn)

    # 计算基本指标
    epsilon = 1e-7
    precision = total_tp / (total_tp + total_fp + epsilon)
    recall = total_tp / (total_tp + total_fn + epsilon)
    f1 = 2 * precision * recall / (precision + recall + epsilon)
    accuracy = total_tp / (total_tp + total_fp + total_fn + epsilon)

    # 计算详细指标
    # 1. 定位误差分析
    avg_shift = np.mean(dist_errors) if len(dist_errors) > 0 else 0
    median_shift = np.median(dist_errors) if len(dist_errors) > 0 else 0
    
    # 定位误差分布
    error_bins = [0, 5, 10, 15, 20, 25, 30]
    error_distribution = {bin_val: 0 for bin_val in error_bins}
    for error in dist_errors:
        for bin_val in sorted(error_bins, reverse=True):
            if error <= bin_val:
                error_distribution[bin_val] += 1
                break

    # 2. 误差类型分析
    false_positive_rate = total_fp / (total_tp + total_fp + epsilon) if (total_tp + total_fp) > 0 else 0
    false_negative_rate = total_fn / (total_tp + total_fn + epsilon) if (total_tp + total_fn) > 0 else 0

    # 3. 性能稳定性分析
    f1_scores = [m["f1"] for m in per_sample_metrics]
    f1_mean = np.mean(f1_scores) if f1_scores else 0
    f1_std = np.std(f1_scores) if f1_scores else 0
    f1_cv = f1_std / f1_mean if f1_mean > 0 else 0

    # 4. 临床相关指标
    # 计算总样本数和总非R波数
    total_samples = batch_size * probs.shape[1]
    total_non_rpeaks = total_samples - (total_tp + total_fn)
    true_negatives = total_non_rpeaks - total_fp
    
    sensitivity = recall  # 敏感性与召回率相同
    specificity = true_negatives / (total_non_rpeaks + epsilon) if total_non_rpeaks > 0 else 0
    positive_predictive_value = precision  # 阳性预测值与精确率相同
    negative_predictive_value = true_negatives / (true_negatives + total_fn + epsilon) if (true_negatives + total_fn) > 0 else 0

    return {
        # 基本指标
        "acc_accuracy": accuracy,
        "acc_precision": precision,
        "acc_recall": recall,
        "acc_f1": f1,
        
        # 定位误差分析
        "avg_shift": avg_shift,
        "median_shift": median_shift,
        "error_distribution": error_distribution,
        
        # 误差类型分析
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "false_positives_count": total_fp,
        "false_negatives_count": total_fn,
        
        # 性能稳定性分析
        "f1_mean": f1_mean,
        "f1_std": f1_std,
        "f1_cv": f1_cv,
        "per_sample_metrics": per_sample_metrics,
        
        # 临床相关指标
        "sensitivity": sensitivity,
        "specificity": specificity,
        "positive_predictive_value": positive_predictive_value,
        "negative_predictive_value": negative_predictive_value,
        
        # 原始计数
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "total_tn": true_negatives
    }