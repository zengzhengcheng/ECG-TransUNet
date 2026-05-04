import os
from pathlib import Path
import numpy as np
import pandas as pd
import random

from sympy import false
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import h5py
from utils import xiaobo,lvbo
from scipy.signal import butter, filtfilt
import torch
import torch.nn.functional as F
import pywt
from ECGDataAugmentation import ECGDataAugmentation
class ECGDataset(Dataset):
    def __init__(self, data_dir, window_size=307200, stride=153600,
                 test_size=0.2, expanded=True,random_seed=42,wave=False):
        """
        初始化 ECG 数据集类
        :param data_dir: 数据文件夹路径
        :param window_size: 每次返回的窗口大小（默认 10 分钟，307200 个数据点）
        :param stride: 滑动窗口步长（默认 5 分钟，153600 个数据点）
        :param test_size: 测试集比例
        :param random_seed: 随机种子，用于划分训练集和测试集
        """
        self.data_dir = data_dir
        self.window_size = window_size
        self.stride = stride
        self.test_size = test_size
        self.random_seed = random_seed
        files=[]
        for path in Path(data_dir).rglob('*.csv'):
            if path.is_file():
                files.append(path)
        self.files = [str(file) for file in files]
        # self.files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]  # 假设数据是 CSV 格式
        self.hdf_dir = os.path.join(data_dir, 'hdf')  # HDF 文件夹路径
        os.makedirs(self.hdf_dir, exist_ok=True)  # 创建 HDF 文件夹（如果不存在）
        self.file_indices = []  # 存储每个文件的索引信息
        self.file_data = {}  # 存储每个文件的 HDF5 文件路径
        self.load_data()
        self.split_data()
        self._build_index_map()
    def _double_diff(self, signal):
        """双差分运算增强QRS波群特征（网页1核心算法）"""
        diff1 = np.diff(signal, n=1)
        diff2 = np.diff(diff1, n=1)
        return np.pad(diff2, (1, 1), 'edge')  # 保持长度一致


    def _static_normalize(self, signal):
        """基于硬件量程的静态标准化（您的核心经验）"""
        # 将int16原始数据转为浮点型
        signal = signal.astype(np.float32)
        # 硬件量程标准化（32767对应±10mV）
        signal = (signal / 32767.0) * 100.0
        signal = np.clip(signal, -20, 20)
        # 双差分处理（保持原有特征增强）
        return self._double_diff(signal)
    def convert_csv_to_hdf(self, csv_file):
        """将 CSV 文件转换为 HDF5 格式"""
        # 获取 CSV 文件路径和 HDF5 文件路径
        filename=os.path.basename(csv_file)
        csv_filepath = csv_file
        hdf_filename =filename.replace('.csv', '.h5')
        hdf_filepath = os.path.join(self.hdf_dir, hdf_filename)

        # 如果 HDF 文件已存在，跳过转换
        if os.path.exists(hdf_filepath):
            return hdf_filepath

        # 读取 CSV 文件并转换为 HDF5 格式
        df = pd.read_csv(csv_filepath)
        # 确保数据是数值类型，如果不是，需要转换
        data = np.array(df['data'].values, dtype=np.float32)
        labels = np.array(df['label'].values, dtype=np.int64)
        # 创建 HDF5 文件并写入数据
        with h5py.File(hdf_filepath, 'w') as f:
            f.create_dataset('data', data=data)
            f.create_dataset('label', data=labels)
        return hdf_filepath
    def load_data(self):
        """读取数据文件并进行预处理"""
        last_end_idx = -1  # 初始化 last_end_idx 用于计算每个文件的 start_idx
        for file in self.files:
            # 检查对应的 HDF 文件是否存在，不存在则进行转换
            hdf_filepath = self.convert_csv_to_hdf(file)

            # 记录文件的 HDF 文件路径
            self.file_data[file] = hdf_filepath

            # 获取该文件的数据和标签的数量
            with h5py.File(hdf_filepath, 'r') as f:
                data = f['data'][:]
                data=data

                # 如果数据长度小于 10 分钟，跳过文件
                if len(data) < self.window_size:
                    print(f"警告：文件 {file} 的数据长度小于 10 分钟，已跳过该文件。")
                    continue

                # 计算该文件可以提供多少个窗口
                num_windows = (len(data) - self.window_size) // self.stride + 1
                # 计算该文件的全局 start_idx 和 end_idx
                start_idx = last_end_idx + 1
                end_idx = start_idx + num_windows - 1
                last_end_idx = end_idx  # 更新 last_end_idx 为当前文件的 end_idx
                # 记录每个文件的全局索引信息
                self.file_indices.append({
                    'file': file,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'num_windows': num_windows
                })
    def split_data(self):
        """根据指定的比例划分训练集和测试集"""
        all_indices = []
        for file_info in self.file_indices:
            start_idx = file_info['start_idx']
            end_idx = file_info['end_idx']
            all_indices.extend(range(start_idx, end_idx + 1))

        # 随机打乱全局索引
        np.random.seed(self.random_seed)
        np.random.shuffle(all_indices)

        # 划分训练集和测试集
        test_size = int(self.test_size * len(all_indices))
        self.test_indices = sorted(all_indices[:test_size])  # 对测试集索引进行排序
        self.train_indices = sorted(all_indices[test_size:])  # 对训练集索引进行排序
    def setTrainStatus(self, train=True):
        """设置训练状态"""
        self.train = train

    def _build_index_map(self):
        """建立 全局索引 -> (文件路径, 文件内窗口起始偏移量) 的映射表"""
        # 使用字典或列表存储，这里用字典映射，key为全局idx
        self.global_to_local_map = {}

        # 遍历所有文件的元数据
        for info in self.file_indices:
            f_path = info['file']
            start_global = info['start_idx']
            end_global = info['end_idx']

            # 对应的每个窗口
            for g_idx in range(start_global, end_global + 1):
                # 计算该窗口在该文件中的第几个步长
                local_window_idx = g_idx - start_global
                # 计算实际的数据点起始位置
                raw_data_start = local_window_idx * self.stride

                self.global_to_local_map[g_idx] = (f_path, raw_data_start)
    def __len__(self):
        """返回数据集的长度"""
        if self.train:
            return len(self.train_indices)
        else:
            return len(self.test_indices)
    def __getitem__(self, idx):
        """
        根据索引返回一个数据样本
        :param idx: 样本索引
        :return: x (ECG 数据), y (标签)
        """
        # 确定全局索引对应的文件和文件内部的索引
        if self.train:
            global_idx = self.train_indices[idx]
        else:
            global_idx = self.test_indices[idx]

        # 查找该索引属于哪个文件
        # file_info = next(f for f in self.file_indices if f['start_idx'] <= global_idx <= f['end_idx'])
        # file = file_info['file']
        # file_data = self.file_data[file]
        # 计算文件内部的索引
        # local_idx = global_idx - file_info['start_idx']
        # start_idx = local_idx * self.stride
        # end_idx = start_idx + self.window_size
        file_path, start_idx = self.global_to_local_map[global_idx]
        file_data_path = self.file_data[file_path]
        end_idx = start_idx + self.window_size
        # 读取数据和标签
        with h5py.File(file_data_path, 'r') as f:
            data = f['data'][start_idx:end_idx]
            data=self._static_normalize(data)
            labels = f['label'][start_idx:end_idx]
            label_map = {0: 0, 1:1,2: 1,3:1, 5: 1, 6: 0}
            # 打印labels，查看是否有未映射的标签
            for label in labels:
                if int(label) not in label_map:
                    print(f"Unmapped label: {int(label)}")
            # 将labels进行转换
            # converted_labels = np.array([label_map[int(label)] for label in labels])
            # converted_labels = converted_labels.astype(np.int64)
            converted_labels = np.array([label_map.get(int(l), 0) for l in labels], dtype=np.int64)
            index = 0
            while index < len(converted_labels):
                if (converted_labels[index] == 1):
                    start = max(0, index - 15)
                    end = min(len(data), index + 15)
                    xdata = data[start:end]
                    maxindex = np.argmax(xdata) + start
                    converted_labels[index] = 0
                    converted_labels[maxindex] = 1
                    index = end + 30
                index += 1
            data=lvbo(data,512)
            data=np.array(data)
            data = xiaobo(data, sample_rate=512, duration=self.window_size // 512)
        return np.array(data, dtype=np.float32),converted_labels,file_path, start_idx
class ECGDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=32, shuffle=True, test=False,num_workers=4):
        """
        自定义 ECG 数据加载器
        :param dataset: ECG 数据集对象
        :param batch_size: 批次大小
        :param shuffle: 是否打乱数据
        :param test: 是否加载测试集数据
        """
        self.test = test
        self.dataset = dataset
        self.dataset.setTrainStatus(train=not test)

        # 使用 DataLoader 初始化
        super().__init__(self.dataset, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers)
class ECG6LabelDataset(ECGDataset):
    def __init__(self, data_dir,expanded=True, window_size=307200, stride=153600,test_size=0.2,random_seed=42,good=1,wave=False):
        self.data_dir = data_dir
        self.window_size = window_size
        self.stride = stride
        self.global_indices = []  # 存储所有包含标签 6 的数据的全局索引
        self.file_local_indices = {}  # 文件局部索引字典
        self.test_indices=[]
        self.train_indices=[]
        self.file_data={}
        self.file_indices={}
        self.test_size = test_size
        self.random_seed = random_seed
        files=[]
        for path in Path(data_dir).rglob('*.csv'):
            if path.is_file():
                files.append(path)
        self.files = [str(file) for file in files]
        self.hdf_dir = os.path.join(data_dir, 'hdf')  # HDF 文件夹路径
        self.good=good
        os.makedirs(self.hdf_dir, exist_ok=True)
        self._build_indices()
        self.split_data()
        self._build_index_map()
    
    def _double_diff(self, signal):
        """双差分运算增强QRS波群特征（网页1核心算法）"""
        diff1 = np.diff(signal, n=1)
        diff2 = np.diff(diff1, n=1)
        return np.pad(diff2, (1, 1), 'edge')  # 保持长度一致

    def _static_normalize(self, signal):
        """基于硬件量程的静态标准化（您的核心经验）"""
        # 将int16原始数据转为浮点型
        signal = signal.astype(np.float32)
        # 硬件量程标准化（32767对应±10mV）
        signal = (signal / 32767.0) * 100.0
        signal = np.clip(signal, -20, 20)
        # 双差分处理（保持原有特征增强）
        return self._double_diff(signal)
    def split_data(self):
        all_indices = list(range(len(self.global_indices)))
        # 随机打乱全局索引
        np.random.seed(self.random_seed)
        np.random.shuffle(all_indices)

        # 划分训练集和测试集
        test_size = int(self.test_size * len(all_indices))
        self.test_indices = sorted(all_indices[:test_size])  # 对测试集索引进行排序
        self.train_indices = sorted(all_indices[test_size:])  # 对训练集索引进行排序

    def setTrainStatus(self, train=True):
        """设置训练状态"""
        self.train = train

    def _build_index_map(self):
        """建立 全局索引 -> (文件路径, 文件内窗口起始偏移量) 的映射表"""
        # 使用字典或列表存储，这里用字典映射，key为全局idx
        self.global_to_local_map = {}

        # 遍历所有文件的元数据
        for filename, (start_global, end_global) in self.file_indices.items():
            # 对应的每个窗口
            for g_idx in range(start_global, end_global + 1):
                if g_idx in self.file_local_indices[filename]:
                    # 计算实际的数据点起始位置
                    raw_data_start = self.file_local_indices[filename][g_idx]
                    self.global_to_local_map[g_idx] = (filename, raw_data_start)

    def __len__(self):
        """返回数据集的长度"""
        if self.train:
            return len(self.train_indices)
        else:
            return len(self.test_indices)
    def convert_csv_to_hdf(self, csv_file):
        """将 CSV 文件转换为 HDF5 格式"""
        # 获取 CSV 文件路径和 HDF5 文件路径
        csv_filepath = os.path.join(self.data_dir, csv_file)
        hdf_filename = csv_file.replace('.csv', '.h5')
        hdf_filepath = os.path.join(self.hdf_dir, hdf_filename)

        # 如果 HDF 文件已存在，跳过转换
        if os.path.exists(hdf_filepath):
            return hdf_filepath

        # 读取 CSV 文件并转换为 HDF5 格式
        df = pd.read_csv(csv_filepath)
        # 确保数据是数值类型，如果不是，需要转换
        data = np.array(df['data'].values, dtype=np.float32)
        labels = np.array(df['label'].values, dtype=np.int64)
        # 创建 HDF5 文件并写入数据
        with h5py.File(hdf_filepath, 'w') as f:
            f.create_dataset('data', data=data)
            f.create_dataset('label', data=labels)
        return hdf_filepath

    def _build_indices(self):
        global_idx = 0
        for file in self.files:
            # 检查对应的 HDF 文件是否存在，不存在则进行转换
            hdf_filepath = self.convert_csv_to_hdf(file)
            # 记录文件的 HDF 文件路径
            self.file_data[file] = hdf_filepath
            with h5py.File(hdf_filepath, 'r') as f:
                data = f['data'][:]
                data=data*10*10
                labels = f['label'][:]

                file_start_global_idx = global_idx  # 记录文件起始全局索引

                for i in range(0, len(data) - self.window_size + 1, self.stride):
                    window_labels = labels[i:i + self.window_size]
                    if 6 in window_labels:
                        self.global_indices.append(global_idx)  # 记录全局索引
                        self.file_local_indices.setdefault(file, {})[global_idx] = i  # 记录文件局部索引
                        global_idx += 1
                    else:
                        if(random.random()>1-self.good):
                            self.global_indices.append(global_idx)  # 记录全局索引
                            self.file_local_indices.setdefault(file, {})[global_idx] = i  # 记录文件局部索引
                            global_idx += 1

                self.file_indices[file] = (file_start_global_idx, global_idx - 1)  # 记录文件全局索引范围
    def __getitem__(self, idx):
        if(self.train):
            global_idx = self.train_indices[idx]
        else:
            global_idx = self.test_indices[idx]
        
        # 使用映射表查找文件和起始位置
        file_path, start_idx = self.global_to_local_map[global_idx]
        file_data_path = self.file_data[file_path]
        end_idx = start_idx + self.window_size
        
        with h5py.File(file_data_path, 'r') as f:
            data = f['data'][start_idx:end_idx]
            data=self._static_normalize(data)
            labels = f['label'][start_idx:end_idx]
            label_map = {0: 0, 1:1,2: 1,3:1, 5: 1, 6: 0}
            # 打印labels，查看是否有未映射的标签
            for label in labels:
                if int(label) not in label_map:
                    print(f"Unmapped label: {int(label)}")
            # 将labels进行转换
            converted_labels = np.array([label_map.get(int(l), 0) for l in labels], dtype=np.int64)
            index = 0
            while index < len(converted_labels):
                if (converted_labels[index] == 1):
                    start = max(0, index - 15)
                    end = min(len(data), index + 15)
                    xdata = data[start:end]
                    maxindex = np.argmax(xdata) + start
                    converted_labels[index] = 0
                    converted_labels[maxindex] = 1
                    index = end + 30
                index += 1
            data=lvbo(data,512)
            data=np.array(data)
            data = xiaobo(data, sample_rate=512, duration=self.window_size // 512)
        return np.array(data, dtype=np.float32),converted_labels,file_path, start_idx


