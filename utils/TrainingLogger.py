import torch
import numpy as np
import pandas as pd
import os
import time
from scipy.signal import find_peaks
from .MetricCalculator import MetricCalculator


class TrainingLogger:
    def __init__(self, log_dir, train_name, heavy_mode=False, buffer_size=50):
        self.log_dir = log_dir
        self.heavy_mode = heavy_mode
        self.train_name = train_name
        self.buffer_size = buffer_size

        # 1. 生成统一的启动时间戳
        start_time = time.strftime("%Y%m%d_%H%M%S")

        # 2. 缓存区（分 mode 存放主日志，难例统一存放）
        self.buffers = {"train": [], "test": []}
        self.hard_buffer = []

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # 3. 定义文件路径 (所有文件共用一个 start_time)
        self.train_csv_path = os.path.join(log_dir, f"{train_name}_train_log_{start_time}.csv")
        self.test_csv_path = os.path.join(log_dir, f"{train_name}_test_log_{start_time}.csv")
        self.hard_mining_path = os.path.join(log_dir, f"{train_name}_hard_examples_{start_time}.csv")

        # 初始化难例文件表头
        if not os.path.exists(self.hard_mining_path):
            with open(self.hard_mining_path, 'w') as f:
                # 增加了 split 列，用于区分是 train 还是 test 的难例
                f.write("split,epoch,global_step,loss,file_path,start_index\n")

        self.attn_dir = os.path.join(log_dir, "attention_maps")
        if not os.path.exists(self.attn_dir):
            os.makedirs(self.attn_dir)

        print(f"[Logger] Initialized. Files timestamp: {start_time}")

    def log_step(self, mode, epoch, global_step, step, index, lr,
                 losses_dict, predictions, targets,
                 metadata=None, model=None):

        record = {"timestamp":time.strftime("%H:%M:%S"),
                  "epoch": epoch, "global_step": global_step,
                  "step": step, "index": index,
                  "lr": lr if mode == 'train' else 0}
        for k, v in losses_dict.items():
            record[f"loss_{k}"] = v if not isinstance(v, torch.Tensor) else v.item()

        with torch.no_grad():
            # --- 1. 计算基础指标 ---
            pred_b = predictions['pred_r'].detach()
            target_mask = targets['pred_r'].detach()

            # 假设 MetricCalculator 已定义
            seg_metrics = MetricCalculator.calc_segmentation_metrics(pred_b, target_mask)
            record.update(seg_metrics)

            # --- 2. 重量级指标 ---
            if self.heavy_mode:
                # 示例：record["metric_peak_shift_ms"] = ...
                if model is not None:
                    total_norm = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            total_norm += p.grad.data.norm(2).item() ** 2
                    record["sys_grad_norm"] = total_norm ** 0.5

            # --- 3. 难例挖掘 (Hard Mining) 逻辑优化 ---
            if metadata is not None:
                # metadata 预期结构: (paths, start_indices, sample_losses) 或 (paths, start_indices, sample_losses, scores)
                if len(metadata) == 4:
                    paths, starts, sample_losses, scores = metadata
                    # 将scores添加到record中
                    if len(scores) > 0:
                        record['score'] = scores[0].item() if isinstance(scores[0], torch.Tensor) else scores[0]
                else:
                    paths, starts, sample_losses = metadata
                if isinstance(sample_losses, torch.Tensor):
                    sample_losses = sample_losses.cpu().numpy()

                # 判定为难例的条件：Loss > 0.5 或者取 Batch 内 Loss 最大的
                hard_indices = np.where(sample_losses > 0.5)[0]
                if len(hard_indices) == 0:
                    hard_indices = [np.argmax(sample_losses)]

                for idx in hard_indices:
                    if idx < len(paths):
                        p = paths[idx]
                        s = starts[idx].item() if isinstance(starts[idx], torch.Tensor) else starts[idx]
                        l = sample_losses[idx]
                        # 将难例行拼成 CSV 格式的字符串，放入缓存
                        line = f"{mode},{epoch},{global_step},{l:.4f},{p},{s}\n"
                        self.hard_buffer.append(line)
        # 写入主日志缓存
        self.buffers[mode].append(record)

        # 满足一定数量写入磁盘
        if len(self.buffers[mode]) >= self.buffer_size:
            self.flush(mode)

        if len(self.hard_buffer) >= self.buffer_size:
            self.flush_hard()

        return record

        # 满足一定数量写入磁盘
        if len(self.buffers[mode]) >= self.buffer_size:
            self.flush(mode)

        if len(self.hard_buffer) >= self.buffer_size:
            self.flush_hard()

        return record

    def flush(self, mode):
        """写入 Train 或 Test 指标日志"""
        if not self.buffers[mode]:
            return

        path = self.train_csv_path if mode == 'train' else self.test_csv_path
        df = pd.DataFrame(self.buffers[mode])
        write_header = not os.path.exists(path)
        df.to_csv(path, mode='a', index=False, header=write_header)
        self.buffers[mode] = []

    def flush_hard(self):
        """将难例缓存写入同一个 CSV"""
        if self.hard_buffer:
            # 持续追加到同一个 hard_mining_path
            with open(self.hard_mining_path, 'a') as f:
                f.writelines(self.hard_buffer)
            self.hard_buffer = []

    def close(self):
        """训练彻底结束时调用，清空所有残留缓存"""
        self.flush('train')
        self.flush('test')
        self.flush_hard()
        print(f"[Logger] All records saved for {self.train_name}")