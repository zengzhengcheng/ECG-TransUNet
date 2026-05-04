import os
import torch
import random
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import pywt
from scipy import signal
from utils.MetricCalculator import MetricCalculator
from utils.TrainingLogger import TrainingLogger
from utils.ECGFeatureExtractor import ECGFeatureExtractor
from utils.AttentionRecorder import AttentionRecorder
def binary_accuracy(preds, y_true):
    # 将预测值通过Sigmoid转换为概率，并取阈值0.5
    y_pred = (torch.sigmoid(preds) >= 0.5).float()
    # 统计正确预测的数量
    correct = (y_pred == y_true).float().sum()
    # 计算准确率
    acc = correct / y_true.shape[0]
    return acc.item()
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
def get_HR(labels,sample_rate):
    data=[]
    for batch in range(len(labels)):
        total=0
        label=labels[batch]
        d=0
        for c in label:
            if c in [1,2]:
                if d != 0:
                    continue
                total+=1
                d=1
            else:
                d=0
        data.extend([total/(len(label)/sample_rate)])
    data=np.array(data)
    return data
def visualize_and_save(data, labels, pred_r_peak, epoch,step, img_dir,mode):
    print("start visualization")
    # 选择最后30秒的数据进行可视化
    start_idx = 0  # 选择最后30秒的数据
    data_segment = data[0, 0,start_idx:start_idx + 153600]
    data_segment=data_segment*100
    labels_segment = labels[0, start_idx:start_idx + 153600]
    pred_segment = pred_r_peak[0, start_idx:start_idx + 153600]
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
    plt.title("Predicted Labels")

    # 保存图片
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    plt.tight_layout()
    plt.savefig(f'{img_dir}/{mode}_Epoch{epoch}_Step{step}.png')
    plt.close()
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

def save_model(model, epoch, step, save_dir='./models'):
    """
    Save the PyTorch model at a specific epoch and step.
    :param model: The model to be saved
    :param epoch: The current epoch number
    :param step: The current step number
    :param save_dir: Directory where the model will be saved
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, f'model_epoch_{epoch}_step_{step}.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Model saved at {save_path}")
def add_subtract_data(ecgdata):
    """
    随机加减 ECG 数据的值
    :param ecgdata: ECG 数据（列表或张量）
    :return: 增强后的 ECG 数据
    """
    if random.random() > 0.5:
        ecgdata += random.uniform(0, 1)
    else:
        ecgdata -= random.uniform(0, 1)
    return ecgdata

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
def lvbo(x,iSampleRate):
    # 1. 带通滤波：使用 filtfilt 替代 lfilter
    b, a = butterBandPassFilter(3, 70, iSampleRate, order=4)
    # x = signal.lfilter(b, a, x)  <-- 删除这一行 (会有延迟)
    x = signal.filtfilt(b, a, x)  # <-- 换成这一行 (零相位，无延迟)

    # 2. 带阻滤波 (50Hz工频干扰)：同样使用 filtfilt
    b, a = butterBandStopFilter(48, 52, iSampleRate, order=2)
    # x = signal.lfilter(b, a, x)  <-- 删除这一行
    x = signal.filtfilt(b, a, x)  # <-- 换成这一行

    # 小波滤波通常相位偏移很小，可以保留
    # 如果 wavelet_filter_ecg 内部用了 shift 相关的操作要注意，但通常 db4 是没问题的
    wavelet_filtered_signal = wavelet_filter_ecg(x, wavelet_name='db4', wavelet_level=4,
                                                 threshold_method='硬阈值')
    return wavelet_filtered_signal
