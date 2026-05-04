import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from datetime import datetime
from models import ECG_SE_ResUNet
from newmodel import AdvancedECGUNet
from transmodel import ECG_TransUNet
from utils import *
import configparser
from ECGDataset import  ECG6LabelDataset,ECGDataLoader,ECGDataLoader,ECGDataAugmentation
import torch
import torch.nn as nn
# 模型定义
class ECGDetectionSystem(nn.Module):
    def __init__(self,input_length=15360,
            in_channels=1,out_channels=1,model=""):
        super().__init__()
        if(model == "res"):
            self.unet = ECG_SE_ResUNet(in_channels=in_channels, n_classes=out_channels)
        elif(model=="adc"):
            self.unet=AdvancedECGUNet(in_channels=in_channels, n_classes=out_channels)
        elif(model=="tra"):
            self.unet=ECG_TransUNet(in_channels=in_channels, out_channels=out_channels)
    def save_weights(self, filepath, epochs, step, prefix=''):
        os.makedirs(filepath, exist_ok=True)

        # 构建包含所有信息的字典
        state_dict = {
            'model_state_dict': self.state_dict(),
            'epochs': epochs,
            'step': step,
        }
        if prefix:
            filename=f"{prefix}_model_{epochs}_{step}.pth"
        else:
            filename=f"model_{epochs}_{step}.pth"
        save_path = os.path.join(filepath, filename)
        # 保存到指定路径
        torch.save(state_dict, save_path)

    def load_weights(self, filepath, device=None, strict=True):
        """
        加载模型权重及相关信息的方法
        参数:
            filepath (str): 权重文件路径
            device (torch.device): 指定加载设备（默认自动检测）
            strict (bool): 是否严格匹配模型参数结构
        返回:
            (epoch, step): 恢复的训练进度信息
        """
        checkpoint = torch.load(filepath, map_location=device)
        try:
            self.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            print(f"[Registry] Load failed: {e}")
        # 解包训练进度
        loaded_epoch = checkpoint['epochs']
        loaded_step = checkpoint['step']

        # 打印恢复信息
        print(f"成功加载权重：{filepath}")
        print(f"训练进度：第 {loaded_epoch} 个 epoch，第 {loaded_step} 个 step")

        return loaded_epoch, loaded_step


    def forward(self, x):
        # 统一前向传播路径
        raw_pred = self.unet(x)
        return raw_pred
