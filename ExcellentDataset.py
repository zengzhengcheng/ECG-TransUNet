import h5py
import numpy as np
from torch.utils.data import Dataset

class ExcellentDataDataset(Dataset):
    """加载excellent_data.hdf文件的数据集"""
    def __init__(self, hdf_path, window_size=15360, stride=7680, test_size=0.1, random_seed=42):
        self.hdf_path = hdf_path
        self.window_size = window_size
        self.stride = stride
        self.test_size = test_size
        self.random_seed = random_seed
        
        # 加载HDF文件
        with h5py.File(hdf_path, 'r') as f:
            # 获取所有文件ID
            self.file_ids = list(f.keys())
            print(f"Found {len(self.file_ids)} files in excellent_data.hdf")
            
            # 收集所有数据
            self.all_data = []
            self.all_labels = []
            self.all_scores = []
            
            for file_id in self.file_ids:
                file_data = f[file_id]
                if 'ecg' in file_data and 'label' in file_data:
                    ecg_data = file_data['ecg'][:]
                    label_data = file_data['label'][:]
                    
                    # 检查数据长度是否匹配窗口大小
                    if len(ecg_data) == window_size and 'quality' in file_data.attrs:
                        self.all_data.append(ecg_data)
                        self.all_labels.append(label_data)
                        # 读取质量分数，直接使用原始值
                        quality = file_data.attrs['quality']
                        
                        # 确保quality是数值
                        quality = float(quality)
                        
                        self.all_scores.append(quality)
            
            print(f"Total windows: {len(self.all_data)}")
            
            # 划分训练集和测试集
            np.random.seed(random_seed)
            indices = np.arange(len(self.all_data))
            np.random.shuffle(indices)
            
            split_idx = int(len(indices) * (1 - test_size))
            self.train_indices = indices[:split_idx]
            self.test_indices = indices[split_idx:]
            
            print(f"Training samples: {len(self.train_indices)}, Test samples: {len(self.test_indices)}")
    
    def __len__(self):
        return len(self.train_indices) if self.train else len(self.test_indices)
    
    def __getitem__(self, idx):
        actual_idx = self.train_indices[idx] if self.train else self.test_indices[idx]
        
        data = self.all_data[actual_idx]
        labels = self.all_labels[actual_idx]
        score = self.all_scores[actual_idx]
        
        return data, labels, f"sample_{actual_idx}", 0, score
    
    def setTrainStatus(self, train=True):
        self.train = train
