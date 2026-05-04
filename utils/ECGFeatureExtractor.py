import os
import torch
import random
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import pywt
from scipy import signal
class ECGFeatureExtractor:
    @staticmethod
    def extract_wavelet_features(ecg_data, wavelet='db4', level=2,tensor=False,device='cpu'):
        """
        稳健的 SWT 特征提取函数。
        自动处理各种维度，并防御性处理 pywt 返回格式问题。
        """
        # 1. 转 Numpy
        if isinstance(ecg_data, torch.Tensor):
            ecg_data = ecg_data.cpu().detach().numpy()

        # 2. 维度处理：统一转为 (Batch, Length)
        # 即使输入是 (L,) 也会变成 (1, L)
        if ecg_data.ndim == 1:
            inputs = ecg_data[np.newaxis, :]
            is_single_sample = True
        elif ecg_data.ndim == 3:
            # (Batch, 1, L) -> (Batch, L)
            inputs = ecg_data.reshape(ecg_data.shape[0], -1)
            is_single_sample = False
        elif ecg_data.ndim == 2:
            inputs = ecg_data
            is_single_sample = False
        else:
            # 极端的 Fallback: 强行展平后按照最后一维长度重组
            # 假设最后一维是 15360
            length = ecg_data.shape[-1]
            inputs = ecg_data.reshape(-1, length)
            is_single_sample = False

        batch_size = inputs.shape[0]
        seq_len = inputs.shape[-1]
        batch_results = []

        # 3. 循环处理
        for i in range(batch_size):
            # 【关键修改1】强制展平为纯一维数组 (Length,)
            # 这能解决 (Length, 1) 导致的各种诡异问题
            signal = inputs[i].flatten()

            try:
                # 【关键修改2】显式指定 trim_approx=False
                # 确保返回格式为 [(cA2, cD2), (cA1, cD1)]
                coeffs = pywt.swt(signal, wavelet, level=level, trim_approx=False)

                # --- 防御性解包逻辑 ---
                # 检查 coeffs[0] 到底是什么
                if isinstance(coeffs[0], (tuple, list)) and len(coeffs[0]) == 2:
                    # 标准情况: [(cA2, cD2), ...]
                    cA2, cD2 = coeffs[0]
                    cA1, cD1 = coeffs[1]
                else:
                    # 异常情况处理 (比如 coeffs 变成扁平列表了)
                    # 如果 trim_approx=True，coeffs 结构是 [cA2, cD2, cD1]
                    # 我们尝试手动分配
                    if len(coeffs) >= 3:
                        cA2 = coeffs[0]
                        cD1 = coeffs[-1]  # 最后一个通常是最高频细节
                    else:
                        raise ValueError(f"Unexpected pywt output format: type={type(coeffs[0])}, len={len(coeffs)}")

                # 4. 标准化
                def normalize(arr):
                    std = np.std(arr)
                    if std < 1e-6: return arr - np.mean(arr)
                    return (arr - np.mean(arr)) / std

                feat_low = normalize(cA2)
                feat_high = normalize(cD1)

                # 堆叠 (2, L)
                sample_feat = np.stack([feat_low, feat_high], axis=0)
                batch_results.append(sample_feat)

            except Exception as e:
                print(f"[Wavelet Error] Sample {i} failed: {e}")
                print(f"  -> Input shape was: {signal.shape}")
                # 填充 0 防止中断
                batch_results.append(np.zeros((2, seq_len), dtype=np.float32))

        # 5. 组合结果
        final_output = np.stack(batch_results, axis=0)
        if(tensor):
            final_output=torch.from_numpy(final_output).float().to(device)
        if is_single_sample:
            return final_output[0]

        return final_output