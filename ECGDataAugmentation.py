import torch
import torch.nn.functional as F
import numpy as np
import random
class ECGDataAugmentation:
    """
    数据增强类，提供数据增强的方法
    """
    def __init__(self, augment=False,expanded=False):
        self.augment = augment
        self.expanded=expanded

    def _robust_normalize_target(self, ecg, mask):
        """
        针对原始数据的鲁棒归一化工具。
        目标：让 R 峰的平均高度接近 1.0，并切除过大的噪声。
        ecg: (B, 1, L)
        mask: (B, 1, L) R峰的标签
        """
        # 复制一份，避免修改原数据影响后续流程
        norm_ecg = ecg.clone()

        batch_size = ecg.size(0)

        for b in range(batch_size):
            # 1. 提取当前样本的 ECG 和 Mask
            current_ecg = ecg[b, 0]
            current_mask = mask[b, 0]

            # 2. 找到 R 峰位置的幅度值
            # 这里的 mask 可能是扩展过的，我们取 > 0.5 的部分
            peak_vals = torch.masked_select(current_ecg, current_mask > 0.5)

            # 3. 计算缩放因子 (Scale Factor)
            if peak_vals.numel() > 0:
                # 使用中位数 (Median) 最稳健，防止个别叠加了噪声的R峰拉偏均值
                median_val = torch.median(torch.abs(peak_vals))

                # 防止除以 0 或极小值
                if median_val < 0.1:
                    scale = 1.0  # 信号太弱，可能全是噪声，放弃缩放
                else:
                    scale = median_val
            else:
                # 如果这一段没有 R 峰，就按整体的 95% 分位数来估算，或者不缩放
                # 简单处理：取最大值的 0.5 倍作为参考
                scale = torch.max(torch.abs(current_ecg)) * 0.5 + 1e-6

            # 4. 执行缩放：让 R 峰接近 1.0
            norm_ecg[b] = current_ecg / scale

        # 5. 硬截断 (Hard Clip)
        # 超过 1.5 的部分，我们认为它就是极值噪声，强行压制
        # 这样 Model A 就不会被迫去拟合那个 20 的大尖刺，而是拟合截断后的平顶 1.5
        # 这对去噪是有益的
        norm_ecg = torch.clamp(norm_ecg, -1.5, 1.5)

        return norm_ecg

    def _generate_median_target(self, ecg, mask):
        """
        生成【患者个性化模板 Target】。
        方法：提取所有心跳 -> 算中位数模板 -> 回填到 Target。
        这能完美去除叠加在 R 峰上的随机噪声，同时保留患者特有的 T 波形态。

        ecg: (B, 1, L)
        mask: (B, 1, L)
        """
        ecg = ecg.clone()
        mask = mask.clone()
        # 1. 准备一个空的 Target
        template_target = torch.zeros_like(ecg)

        # 窗口半径 (覆盖 P-QRS-T)
        # 512Hz 下，0.6秒大约 300点，半径取 150
        radius = 150
        seq_len = ecg.shape[-1]

        batch_size = ecg.shape[0]

        for b in range(batch_size):
            current_ecg = ecg[b, 0]
            current_mask = mask[b, 0]

            # 找到 R 峰位置
            # 使用 > 0.5 过滤扩展过的 mask，找连通域中心或极大值
            # 简单起见，这里假设 mask 已经被 skeletonize 过或者是稀疏的
            # 如果是宽 mask，需要先取中心。这里我们在 loop 里动态找一下峰值

            # 简单的寻峰逻辑 (基于 Mask)
            peak_indices = torch.nonzero(current_mask > 0.5).squeeze()
            if peak_indices.numel() == 0: continue

            # 如果 mask 是连续的宽块，nonzero 会返回一堆连续索引
            # 我们需要去重，找到每个块的中心
            # 这里简单处理：如果索引连续，只取中间的
            # (为代码简洁，这里假设传入的 mask 最好是经过 _skeletonize 的，或者是稀疏的)
            # 如果没有 _skeletonize，下面的逻辑可能需要稍微聚类一下，但通常此时用 label 原始数据里的 1 更准

            # 假设 peak_indices 就是大概的位置
            # 为了严谨，我们将连续索引分组 (cluster)
            diff = peak_indices[1:] - peak_indices[:-1]
            # 找到不连续的点作为分割
            split_points = (diff > 10).nonzero().squeeze() + 1
            if split_points.numel() == 0:
                clusters = [peak_indices]
            else:
                if split_points.dim() == 0: split_points = split_points.unsqueeze(0)
                clusters = torch.tensor_split(peak_indices, split_points.cpu())

            real_peaks = []
            for cluster in clusters:
                if cluster.numel() > 0:
                    real_peaks.append(cluster[cluster.numel() // 2].item())

            # --- 核心逻辑：提取所有拍 ---
            beats_list = []
            valid_peaks = []

            for p in real_peaks:
                # 边界检查
                if p - radius < 0 or p + radius >= seq_len:
                    continue

                # 切片 (窗口长度 301)
                beat_segment = current_ecg[p - radius: p + radius + 1]
                beats_list.append(beat_segment)
                valid_peaks.append(p)

            if not beats_list:
                continue

            # 堆叠所有心跳: (N, 301)
            beats_stack = torch.stack(beats_list)

            # --- 核心逻辑：计算中位数模板 ---
            # dim=0，算出这一条数据的“标准长相”
            median_template, _ = torch.median(beats_stack, dim=0)

            # 归一化模板 (R峰为1)
            # 找到模板里的最大值(R峰)
            # 注意：有时 R 峰不在正中心(因为标签可能有偏差)，需要对齐一下
            # 这里简单做幅度归一化
            max_val = torch.max(torch.abs(median_template))
            if max_val > 0.1:
                median_template = median_template / max_val

            # 硬截断噪声 (防止模板里混入固定的噪声)
            median_template = torch.clamp(median_template, -1.2, 1.2)

            # --- 核心逻辑：回填 Target ---
            for p in valid_peaks:
                template_target[b, 0, p - radius: p + radius + 1] = median_template

        return template_target

    def _generate_adaptive_median_target(self, ecg, mask,device='cpu'):
        """
        【升级版】自适应患者个性化模板生成。
        1. 动态计算窗口大小 (防止高心率重叠)。
        2. 局部自动对齐 R 峰 (防止标签偏差)。
        3. 剔除噪声样本 (基于相关性筛选，防止烂数据污染模板)。
        """
        template_target = torch.zeros_like(ecg)
        batch_size = ecg.shape[0]
        seq_len = ecg.shape[-1]

        # 定义一个安全的最小半径 (例如 0.05秒 -> 25点)，防止过小
        min_radius = 25

        for b in range(batch_size):
            current_ecg = ecg[b, 0]
            current_mask = mask[b, 0]

            # 1. 获取所有 R 峰索引
            # 使用 > 0.5 过滤
            peak_indices = torch.nonzero(current_mask > 0.5).squeeze()
            if peak_indices.numel() < 2:
                continue  # 只有一个心跳没法算RR，也没法算模板，跳过

            # 处理连续索引 (骨架化)，取每个块的中心
            diff = peak_indices[1:] - peak_indices[:-1]
            split_points = (diff > 5).nonzero().squeeze() + 1
            if split_points.numel() == 0:
                clusters = [peak_indices]
            else:
                if split_points.dim() == 0: split_points = split_points.unsqueeze(0)
                clusters = torch.tensor_split(peak_indices, split_points.cpu())

            real_peaks = []
            for cluster in clusters:
                if cluster.numel() > 0:
                    # 简单取中点，后面会精修
                    real_peaks.append(cluster[cluster.numel() // 2].item())

            if len(real_peaks) < 2: continue

            # 2. 【核心优化】计算自适应半径
            # 计算 RR 间隔
            rr_intervals = np.diff(real_peaks)
            # 取 RR 间隔的中位数，除以2，再乘以一个安全系数(0.9)，保证不重叠
            median_rr = np.median(rr_intervals)
            adaptive_radius = int((median_rr / 2) * 0.9)

            # 限制范围：不能小于 25，也不能太大(超过 150)
            radius = max(min_radius, min(150, adaptive_radius))
            kernel_size = 2 * radius + 1

            # 3. 提取切片 & 局部对齐
            beats_list = []
            valid_peaks = []

            for p in real_peaks:
                # 边界检查
                start_raw = p - radius
                end_raw = p + radius + 1
                if start_raw < 0 or end_raw > seq_len:
                    continue

                # --- 局部对齐 (Refinement) ---
                # 在标点附近 +/- 10 个点内找最大值，重新定中心
                # 这能解决标签稍微标偏的问题
                search_radius = 10
                s_search = max(0, p - search_radius)
                e_search = min(seq_len, p + search_radius)
                local_seg = current_ecg[s_search:e_search]
                # 找到局部最大值的相对偏移
                offset = torch.argmax(torch.abs(local_seg)) - (p - s_search)
                corrected_p = p + offset

                # 重新切片
                start = corrected_p - radius
                end = corrected_p + radius + 1
                if start < 0 or end > seq_len: continue

                segment = current_ecg[start:end]
                # 长度必须一致
                if segment.shape[0] != kernel_size: continue

                beats_list.append(segment)
                valid_peaks.append(corrected_p)

            if not beats_list: continue

            beats_stack = torch.stack(beats_list)  # (N, L)

            # 4. 【核心优化】模板清洗 (Template Cleaning)
            # 先算一个粗略的中位数
            rough_median, _ = torch.median(beats_stack, dim=0)

            # 计算每个切片与粗略模板的相关性 (Pearson Correlation)
            # 简单的点乘近似：由于都在一个数量级
            correlations = []
            for i in range(beats_stack.shape[0]):
                # 计算 Cosine Similarity 或者简单的 MSE
                # 这里用 MSE 剔除异常值更直接
                mse = F.mse_loss(beats_stack[i], rough_median)
                correlations.append(mse)

            correlations = torch.stack(correlations)
            # 找出 MSE 最小的 50% 样本 (最像平均脸的那些)
            # 或者设定一个阈值
            cutoff = torch.quantile(correlations, 0.5)  # 只留最好的一半
            good_indices = (correlations <= cutoff).nonzero().squeeze()

            if good_indices.numel() == 0:
                # 如果都很烂，就用粗略模板，或者 fallback 到 ideal generator
                final_template = rough_median
            else:
                # 只用好样本算最终模板
                if good_indices.dim() == 0: good_indices = good_indices.unsqueeze(0)
                clean_beats = beats_stack[good_indices]
                final_template, _ = torch.median(clean_beats, dim=0)

            # 5. 归一化模板
            max_val = torch.max(torch.abs(final_template))
            if max_val > 0.1:
                final_template = final_template / max_val
            final_template = torch.clamp(final_template, -1.2, 1.2)

            # 6. 回填
            for p in valid_peaks:
                start = p - radius
                end = p + radius + 1
                template_target[b, 0, start:end] = final_template
        return template_target.to(device)
    def _apply_masking(self, x, mask_ratio=0.15):
        """
        掩码增强 (Masked Modeling Strategy)
        随机将一段或多段信号置为 0，强迫模型利用上下文(Transformer/LSTM)去"脑补"
        """
        # x shape: (..., Length)
        x=x.clone()
        length = x.shape[-1]

        # 决定掩码长度
        mask_len = int(length * mask_ratio)

        # 随机选择起始位置
        start_idx = torch.randint(0, length - mask_len, (1,)).item()
        end_idx = start_idx + mask_len

        # 将该区域置为 0 (或者均值，这里用0模拟信号丢失)
        # 注意：只修改 Input，Target 依然是干净的完整波形，这样模型就必须学会"无中生有"
        x[..., start_idx:end_idx] = 0.0

        return x

    def _random_local_attenuation(self, x):
        """
        【针对性修复】局部幅度衰减
        模拟图中R波幅度降低的情况。随机选中一段区域，将其幅度乘以 0.2~0.5
        强迫模型学会：即使波很小，只要形状对、节奏对，它就是R波。
        """
        x=x.clone()
        if torch.rand(1) > 0.5:  # 50% 概率触发
            length = x.shape[-1]

            # 衰减窗口长度：1000 到 3000 个点 (约 2-6 秒)
            atten_len = torch.randint(1000, 3000, (1,)).item()
            start_idx = torch.randint(0, max(1, length - atten_len), (1,)).item()
            end_idx = start_idx + atten_len

            # 生成衰减系数 (0.2 到 0.6)
            scale = 0.2 + 0.4 * torch.rand(1).item()

            # 应用衰减
            x[..., start_idx:end_idx] *= scale

        return x

    def _add_grunting_noise(self, x):
        """添加猪呼噜声模拟噪声"""
        # 生成随机基频 (2-5Hz)
        # freq = 2 + 3 * torch.rand(1).item()  # 使用.item()转为Python标量

        # # 生成正弦噪声
        # num_samples = x.size(-1)  # 获取信号长度
        # t = torch.linspace(0, 2 * np.pi * freq, num_samples, device=x.device)
        # noise = 0.3 * torch.sin(t)

        # return x + noise.to(x.device)
        # 复制输入张量以避免修改原始数据
        x = x.clone()
        
        # 计算信号标准差用于噪声幅度缩放
        x_std = x.std().detach()
        
        # 随机决定是否添加呼噜声噪声 (70% 概率)
        if torch.rand(1) < 0.7:
            batch_size, channels, seq_len = x.shape
            
            # 为每个样本生成间歇性的呼噜声
            for b in range(batch_size):
                # 随机生成多个呼噜声事件
                num_events = torch.randint(1, 4, (1,)).item()  # 1-3个事件
                
                for _ in range(num_events):
                    # 生成随机基频 (2-5Hz)
                    freq = 2 + 3 * torch.rand(1).item()
                    
                    # 生成随机持续时间 (0.5-2秒)
                    duration = int(torch.randint(512, 2048, (1,)).item())  # 1-4秒 @ 512Hz
                    
                    # 随机起始位置
                    start_idx = torch.randint(0, seq_len - duration, (1,)).item()
                    end_idx = start_idx + duration
                    
                    # 生成正弦噪声
                    t = torch.linspace(0, 2 * np.pi * freq * (duration / 512), duration, device=x.device)
                    noise = 0.3 * x_std * torch.sin(t)
                    
                    # 添加噪声到信号
                    x[b, :, start_idx:end_idx] += noise
        
        return x
    def apply_chaos(self,ecgdata,labels):
        ecgdata = ecgdata.clone()
        if random.random() > 0.7:
            batch_size = ecgdata.shape[0]
            seq_len = ecgdata.shape[-1]

            for b in range(batch_size):
                # 随机选择一种噪声模式
                # 'spike': 带动周围点的尖峰
                # 'step':  持续的高低值
                # 'chaos': 乱序值
                noise_mode = random.choice(['spike', 'step', 'chaos'])

                # 随机生成噪声幅度，要求大于 20
                amplitude = random.uniform(20, 50)
                if random.random() > 0.5: amplitude *= -1  # 随机正负

                # 随机起始位置
                start_idx = random.randint(0, seq_len - 150)  # 留出余量

                # --- 模式 A: 尖峰 (带动周围点变大) ---
                if noise_mode == 'spike':
                    # 宽度 5-20 个点
                    width = random.randint(20, 40)
                    end_idx = min(start_idx + width, seq_len)

                    # 使用切片操作添加一个“矩形波”或“三角波”
                    # 这里直接加矩形波最简单，模拟剧烈突变
                    if ecgdata.dim() == 3:  # (Batch, Channel, Length)
                        ecgdata[b, :, start_idx:end_idx] += amplitude
                    else:  # (Batch, Length)
                        ecgdata[b, start_idx:end_idx] += amplitude

                # --- 模式 B: 持续高低值 (饱和/接触不良) ---
                elif noise_mode == 'step':
                    # 持续时间 50-200 个点
                    duration = random.randint(50, 200)
                    end_idx = min(start_idx + duration, seq_len)

                    if ecgdata.dim() == 3:
                        ecgdata[b, :, start_idx:end_idx] += amplitude
                    else:
                        ecgdata[b, start_idx:end_idx] += amplitude

                # --- 模式 C: 乱序值 (剧烈肌电干扰/设备故障) ---
                elif noise_mode == 'chaos':
                    # 持续时间 20-100 个点
                    duration = random.randint(40, 400)
                    end_idx = min(start_idx + duration, seq_len)

                    # 生成同形状的随机噪声
                    target_slice = ecgdata[b, ..., start_idx:end_idx]
                    # randn 产生标准正态分布，乘以 amplitude 放大
                    noise = torch.randn_like(target_slice) * abs(amplitude)

                    ecgdata[b, ..., start_idx:end_idx] += noise
        return ecgdata,labels

    def _add_bedding_artifacts(self, x):
        x = x.clone()
        x_std = x.std().detach()
        # 突发高频干扰
        if torch.rand(1) < 0.3:
            burst = torch.randn_like(x) * (x_std / 2)

            # 修改后的池化参数
            burst = F.max_pool1d(
                burst.abs(),
                kernel_size=5,
                stride=1,  # 步长设为1
                padding=2  # 填充保证长度不变
            )

            x += burst
        if torch.rand(1) < 0.05:  # 独立触发概率
            # 生成高强度脉冲干扰（方法1：随机尖峰）
            spike_amplitude = 15 + 200 * torch.rand(1)  # 幅度范围 [500, 1000]
            spike = torch.zeros_like(x)
            spike_length = 40  # 尖峰持续时间（时间步数）
            start_idx = torch.randint(0, x.shape[-1] - spike_length, (1,))
            spike[..., start_idx:start_idx + spike_length] = spike_amplitude

            x += spike
        if torch.rand(1) < 0.2:
            # 随机生成噪声长度（30~100个时间步）
            noise_length = torch.randint(30, 100, (1,)).item()
            signal_length = x.shape[-1]

            # 随机选择噪声起始位置（确保不越界）
            start_idx = torch.randint(0, signal_length - noise_length, (1,)).item()
            end_idx = start_idx + noise_length

            # 生成局部高斯噪声（强度为原始信号标准差的1~3倍）
            noise_amplitude = x_std * (1 + 2 * torch.rand(1, device=x.device))  # 1x~3x标准差
            local_noise = torch.randn_like(x[..., start_idx:end_idx]) * noise_amplitude

            # 将噪声叠加到原始信号
            x[..., start_idx:end_idx] += local_noise
        return x
    def expanded_label(self,labels,expansion_range=5):
        labels = labels.copy()
        original_shape = labels.shape
        work_labels = labels.reshape(-1, labels.shape[-1])

        # 3. 创建副本进行操作，不影响原数据
        expanded_labels = np.copy(work_labels)

        if self.expanded:
            seq_len = work_labels.shape[-1]
            num_samples = work_labels.shape[0]

            # 遍历每一条数据（Batch 中的每一个样本）
            for b in range(num_samples):
                # 优化：只获取值为 1 的索引，而不是遍历所有点 (大幅提升速度)
                # np.where 返回的是 tuple，取 [0] 得到索引数组
                ones_indices = np.where(work_labels[b] == 1)[0]

                for idx in ones_indices:
                    # 计算左右扩展的边界，并限制在 [0, seq_len] 范围内防止越界
                    start = max(0, idx - expansion_range)
                    end = min(seq_len, idx + expansion_range + 1)

                    # 使用切片赋值 (比循环一个个赋值快得多)
                    expanded_labels[b, start:end] = 1

        # 4. 恢复为原始形状并返回
        return expanded_labels.reshape(original_shape)

    def expanded_probility_label(self, labels, expansion_range=5, mode='linear',tensor=False,device='cpu'):
        """
        标签软扩展：越靠近中心，概率越高。
        :param labels: 原始标签
        :param expansion_range: 单侧半径
        :param mode: 'hard' (全1), 'linear' (线性衰减), 'gaussian' (高斯)
        """
        if(isinstance(labels,np.ndarray)):
            labels = labels.copy()
        if isinstance(labels, torch.Tensor):
            labels = labels.clone().cpu()
        original_shape = labels.shape
        work_labels = labels.reshape(-1, labels.shape[-1])

        # 创建一个全 0 的底板，浮点型（因为要有 0.8 这种小数）
        expanded_labels = np.zeros_like(work_labels, dtype=np.float32)

        if self.expanded:
            seq_len = work_labels.shape[-1]
            num_samples = work_labels.shape[0]

            # --- 1. 预先生成一个“印章” (Kernel) ---
            # 比如 range=2: [0.6, 0.8, 1.0, 0.8, 0.6]
            if mode == 'linear':
                # 衰减步长
                step = 1.0 / (expansion_range + 2)  # 稍微除大一点，保证边缘不为0
                kernel_side = np.linspace(1.0 - step, 0.1, expansion_range)
                kernel = np.concatenate([kernel_side[::-1], [1.0], kernel_side])
                # 简单的截断防止负数（虽然上面的逻辑保证了正数）
                kernel = np.clip(kernel, 0, 1)
            elif mode == 'hard':
                kernel = np.ones(2 * expansion_range + 1)

            kernel_len = len(kernel)
            radius = expansion_range

            # --- 2. 盖章 ---
            for b in range(num_samples):
                # 找到 R 峰位置
                ones_indices = np.where(work_labels[b] == 1)[0]

                for idx in ones_indices:
                    # 计算左右边界
                    start_idx = idx - radius
                    end_idx = idx + radius + 1

                    # 计算 Kernel 的截取范围 (处理越界情况)
                    k_start = 0
                    k_end = kernel_len

                    # 左越界处理
                    if start_idx < 0:
                        k_start = -start_idx
                        start_idx = 0

                    # 右越界处理
                    if end_idx > seq_len:
                        k_end = kernel_len - (end_idx - seq_len)
                        end_idx = seq_len

                    # 有效区域
                    if end_idx > start_idx:
                        # 【关键】使用 np.maximum 取最大值
                        # 如果两个心跳离得近，0.6 和 0.8 重叠，取 0.8，而不是相加
                        expanded_labels[b, start_idx:end_idx] = np.maximum(
                            expanded_labels[b, start_idx:end_idx],
                            kernel[k_start:k_end]
                        )
        expanded_labels=expanded_labels.reshape(original_shape)
        if(tensor):
            expanded_labels=torch.from_numpy(expanded_labels).float().to(device)
        return expanded_labels.reshape(original_shape)
    def apply_clean_augmentation(self, ecgdata):
        # 2. 掩码增强 (Masking)
        # 建议概率设低一点 (e.g., 20%)，因为这个任务很难
        ecgdata = ecgdata.clone()
        if random.random() < 0.2:
            ecgdata = self._apply_masking(ecgdata)
        # 方法3: 数据缩放（1 到 1.2 倍）
        if random.random() > 0.5:
            scale = random.uniform(0.8, 1.2)  # 可以缩小至 0.8 倍
            ecgdata *= scale
        return ecgdata

    def apply_augmentation(self, ecgdata):
        """
        应用数据增强
        :param ecgdata: ECG数据
        :param labels: 标签数据
        :return: 增强后的数据和标签
        """
        ecgdata = ecgdata.clone()
        if not self.augment:
            return ecgdata

        # 方法1: 数据加减（加或者减）
        ecgdata=self._add_grunting_noise(ecgdata)  # 呼噜声干扰
        ecgdata=self._add_bedding_artifacts(ecgdata)  # 呼噜声干扰
        ecgdata = self._random_local_attenuation(ecgdata)

        # 2. 掩码增强 (Masking)
        # 建议概率设低一点 (e.g., 20%)，因为这个任务很难
        if random.random() < 0.2:
            ecgdata = self._apply_masking(ecgdata)
        # 方法3: 数据缩放（1 到 1.2 倍）
        if random.random() > 0.5:
            scale = random.uniform(0.5, 2.2)  # 可以缩小至 0.8 倍
            ecgdata *= scale
        return ecgdata