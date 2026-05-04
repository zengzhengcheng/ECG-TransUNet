import torch
import numpy as np
import pandas as pd
import os
import time
from scipy.signal import find_peaks


class MetricCalculator:
    """
    静态工具类：负责计算各种科研指标。
    """

    @staticmethod
    def calc_segmentation_metrics(pred_logits, target_mask, threshold=0.5):
        """
        [轻量级] 计算基础分割指标
        """
        # 转概率并二值化
        probs = torch.sigmoid(pred_logits)
        preds = (probs > threshold).float().view(-1)
        targets = target_mask.float().view(-1)

        # 混淆矩阵元素
        tp = (preds * targets).sum().item()
        fp = ((1 - targets) * preds).sum().item()
        fn = (targets * (1 - preds)).sum().item()
        tn = ((1 - targets) * (1 - preds)).sum().item()

        epsilon = 1e-7

        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)  # Sensitivity
        f1 = 2 * (precision * recall) / (precision + recall + epsilon)  # Dice Coefficient
        iou = tp / (tp + fp + fn + epsilon)
        specificity = tn / (tn + fp + epsilon)

        return {
            "seg_iou": iou,
            "seg_f1": f1,
            "seg_prec": precision,
            "seg_recall": recall,
            "seg_spec": specificity
        }
    @staticmethod
    def calc_reconstruction_metrics(pred_ecg, target_ecg):
        """
        [复用] 计算重建指标 (MSE, PCC, SNR)
        """
        # 确保不计算梯度
        if isinstance(pred_ecg, torch.Tensor): pred_ecg = pred_ecg.detach().cpu()
        if isinstance(target_ecg, torch.Tensor): target_ecg = target_ecg.detach().cpu()

        x = pred_ecg.view(-1)
        y = target_ecg.view(-1)

        mse = torch.mean((x - y) ** 2).item()

        # PCC
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        pcc = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-7)

        # SNR
        signal_power = torch.sum(y ** 2)
        noise_power = torch.sum((x - y) ** 2)
        snr = 10 * torch.log10((signal_power / (noise_power + 1e-7)) + 1e-7)

        return {
            "recon_mse": mse,
            "recon_pcc": pcc.item(),
            "recon_snr": snr.item()
        }

    @staticmethod
    def calc_r_peak_shift_ms(pred_logits, target_mask, fs=512):
        """
        [重量级指标] 计算 R 峰定位的时间偏差 (MAbsE - Mean Absolute Error)。

        :param pred_logits: 模型输出的 Logits (B, L)
        :param target_mask: 真实标签 (B, L) (兼容宽标签或稀疏标签)
        :param fs: 采样率，用于将点数转换为毫秒
        :return: 平均偏移量 (ms)
        """
        # 1. 转换与形状标准化
        if isinstance(pred_logits, torch.Tensor):
            pred_logits = pred_logits.detach().cpu().numpy()
        if isinstance(target_mask, torch.Tensor):
            target_mask = target_mask.detach().cpu().numpy()

        probs = 1.0 / (1.0 + np.exp(-pred_logits))

        # 维度标准化 helper (内部使用)
        def _flatten_batch(arr):
            if arr.ndim == 1: return arr[np.newaxis, :]
            if arr.ndim == 3: return arr.reshape(arr.shape[0], -1)
            return arr

        probs = _flatten_batch(probs)
        target_mask = _flatten_batch(target_mask)

        batch_size = probs.shape[0]
        all_shifts = []

        # 容差窗口：只匹配 100ms 范围内的点，太远的不算偏移，算漏报，不计入此指标
        # 512Hz * 0.1s = 51 points
        match_tolerance_points = int(fs * 0.1)

        for b in range(batch_size):
            # A. 找预测峰 (Height > 0.5, 最小间距防止双峰)
            p_peaks, _ = find_peaks(probs[b], height=0.5, distance=30)

            # B. 找真值峰 (即便是宽标签，find_peaks 也能找到中心极值点)
            # height=0.5 兼容软标签和硬标签
            t_peaks, _ = find_peaks(target_mask[b], height=0.5, distance=30)

            if len(t_peaks) == 0 or len(p_peaks) == 0:
                continue  # 无法计算偏移

            # C. 最近邻匹配
            for p in p_peaks:
                # 计算该预测点到所有真值点的距离
                distances = np.abs(t_peaks - p)
                min_dist = np.min(distances)

                # 只有当距离在合理范围内，才认为是匹配上了，计入偏移
                # 如果距离太远，那是 FP (误报)，不应计入定位精度
                if min_dist <= match_tolerance_points:
                    all_shifts.append(min_dist)

        # 3. 汇总计算
        if len(all_shifts) == 0:
            return 0.0

        # 平均点数误差
        avg_shift_points = np.mean(all_shifts)

        # 转换为毫秒: (点数 / fs) * 1000
        avg_shift_ms = (avg_shift_points / fs) * 1000.0

        return avg_shift_ms
    @staticmethod
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