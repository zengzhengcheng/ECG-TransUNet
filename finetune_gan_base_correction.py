import os.path
import sys
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import h5py
from torch.utils.data import Dataset

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ablistimunet.ECGDataset import ECGDataset, ECGDataLoader, ECG6LabelDataset
from ablistimunet.ECGDataAugmentation import ECGDataAugmentation
from ablistimunet.ExcellentDataset import ExcellentDataDataset
from datetime import datetime
from ablistimunet.gongju import *
import configparser
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from model import ECGDetectionSystem
from ablistimunet.utils import ECGFeatureExtractor
from ablistimunet.utils import TrainingLogger, MetricCalculator
from ablistimunet.losses import AdvancedSegmentationLoss, AdvancedGenerativeLoss, AggressiveSegmentationLoss

# 模型定义
config = configparser.ConfigParser()
# 读取配置文件
config.read('config.ini', encoding="utf-8")
machine_list = ["macmachine", "ubuntumachine", "winmachine"]
dir_list = ["common", "dirganbase1", "dirganbase2", "dirganbase3"]

dir_name = dir_list[3]
machine = machine_list[1]
# 获取配置信息
data_dir = config.get(machine, 'data_dir')  # 从 [DEFAULT] 节获取 data_dir 选项
img_dir = config.get(dir_name, 'img_dir')
img_dir = os.path.join("imgAllC", img_dir)
model_dir = config.get(dir_name, 'model_dir')
model_dir = os.path.join('modelAllC', model_dir)
model_dir = os.path.join(os.getcwd(), model_dir)
ganmodel_dir = os.path.join("./modelAllC", "modelgan1/correct_gan_model_9_3620.pth")
window_stride = config.getint(dir_name, 'window_stride')  # 注意这里使用 getint 获取整数值
useSwanLab = config.getboolean(dir_name, 'useSwanLab')
totallength = 30
baselength = 0
predictlength = totallength - baselength
start_epoch = 0
epochs = 10
sample_rate = 512
window_size = totallength * sample_rate

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available()
                      else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(device)

# 测试函数 - 按照数据分数分别记录
def test_model(model, ganmodel, device, test_loader, logger, epoch, step, img_dir, batch_size=16):
    model.eval()  # 设置为评估模式
    ganmodel.eval()  # GAN模型也设置为评估模式
    all_preds_dict = {}  # 用字典保存每个数据点的预测标签
    pwindow_data = None
    mining_criterion_bce = nn.BCEWithLogitsLoss(reduction='none')
    
    # 按分数记录指标
    score_metrics = {}
    
    with torch.no_grad():  # 测试时不计算梯度
        finalacc = []
        for teststep, (data, labels, paths, start_indices, scores) in enumerate(test_loader):
            allacc = []
            data = data.unsqueeze(1)  # 变成 (batch_size, 1, seq_len)
            num_windows = (data.size(2) - (window_size - window_stride)) // window_stride
            for j in range(num_windows):
                start = j * window_stride
                end = start + window_size  # 每个滑动窗口覆盖7680个数据点
                window_data = data[:, :, start:end]
                window_data = window_data.to(device)
                wavelet_feats_np = ECGFeatureExtractor.extract_wavelet_features(window_data)
                wavelet_tensor = torch.from_numpy(wavelet_feats_np).float().to(device)  # (2, L)
                input_data = torch.cat([
                    window_data,  # Ch 0
                    wavelet_tensor  # Ch 3, 4
                ], dim=1)
                window_labels = labels[:, start:end]
                target_clean_data = ECGSynthesizer(window_labels.to(device))
                target_clean_data = target_clean_data.to(device)
                window_labels = agument.expanded_probility_label(window_labels)
                window_labels = torch.from_numpy(window_labels)
                window_labels = window_labels.to(device).float()
                # 前向传播获取预测
                clean_ecg, pred_r_peak = ganmodel(input_data)
                probabilities = torch.sigmoid(pred_r_peak)
                # 设定阈值（0.5）得到二分类标签
                threshold = 0.5
                pred_labels = (probabilities >= threshold).float()
                pred_labels = pred_labels.squeeze()
                pred_labels = pred_labels.reshape([1, 1, -1])
                
                # 计算GAN模型的准确性
                gan_accuracy_dict = calc_ecg_accuracy_detailed(pred_r_peak, window_labels)
                gan_accuracy = gan_accuracy_dict['acc_precision']
                
                wavelet_feats_np = ECGFeatureExtractor.extract_wavelet_features(clean_ecg.cpu().detach().numpy())
                wavelet_tensor = torch.from_numpy(wavelet_feats_np).float().to(device)  # (2, L)
                input_data = torch.cat([
                    clean_ecg,  # Ch 0
                    wavelet_tensor,  # Ch 3, 4
                    pred_labels
                ], dim=1)
                pred_r_peakgan = pred_labels.cpu().numpy()  # 预测标签
                pred_r_peak = model(input_data)
                probabilities = torch.sigmoid(pred_r_peak)

                # 设定阈值（0.5）得到二分类标签
                threshold = 0.5
                pre_labels = (probabilities >= threshold).float()
                pre_labels = pre_labels.squeeze()
                # 获取预测标签和概率
                pred_labels = pre_labels.cpu().numpy()  # 预测标签
                accuracy_dict = calc_ecg_accuracy_detailed(pred_r_peak, window_labels)
                accuracy = accuracy_dict['acc_precision']
                
                # 合并GAN和base模型的准确性到一个字典中
                combined_accuracy_dict = accuracy_dict.copy()
                combined_accuracy_dict['gan_acc_precision'] = gan_accuracy
                combined_accuracy_dict['gan_acc_recall'] = gan_accuracy_dict['acc_recall']
                combined_accuracy_dict['gan_acc_f1'] = gan_accuracy_dict['acc_f1']
                
                # 按分数记录指标
                for score in scores:
                    # 保留原始分数，允许0.5分的存在
                    score_val = score.item()
                    # 确保分数在0-5范围内
                    score_val = max(0, min(5, score_val))
                    score_str = str(score_val)
                    if score_str not in score_metrics:
                        score_metrics[score_str] = []
                    score_metrics[score_str].append({
                        'precision': accuracy_dict["acc_precision"],
                        'recall': accuracy_dict["acc_recall"],
                        'f1': accuracy_dict["acc_f1"],
                        'gan_precision': gan_accuracy_dict["acc_precision"],
                        'gan_recall': gan_accuracy_dict["acc_recall"],
                        'gan_f1': gan_accuracy_dict["acc_f1"]
                    })
                
                pred_dict = {
                    'clean_ecg': target_clean_data,
                    'pred_r': pred_r_peak,
                    'gan_pred_r': pred_r_peak  # 保存GAN的预测
                }

                # 包装 Targets
                target_dict = {
                    'clean_ecg': clean_ecg,  # 无论是合成还是原始
                    'pred_r': window_labels.float()
                }
                current_lr = 0
                # 一键记录
                with torch.no_grad():
                    # 我们用 Model B 的分割误差来衡量样本难易程度
                    # BCE per pixel: (B, 1, L)
                    pred_r_peak = normalize_input_shape(pred_r_peak)
                    window_labels = normalize_input_shape(window_labels)
                    raw_bce = mining_criterion_bce(pred_r_peak, window_labels)
                    # 加上 weight map (关注困难区域) - 与训练过程保持一致
                    target, weight_map = generate_weight_from_labels(window_labels, is_soft_label=True, to_tensor=True)
                    weight_map = weight_map.to(device)
                    window_labels = window_labels.to(device)
                    weighted_bce = raw_bce * weight_map
                    # 对每个样本求平均 -> (B,)
                    sample_losses = weighted_bce.mean(dim=(1, 2))
                # 将质量分数添加到metadata中
                metadata = (paths, start_indices, sample_losses, scores)
                metrics = logger.log_step(
                    mode='test',
                    epoch=epoch,  # 1. Epoch
                    global_step=step,  # 2. Global Step
                    step=teststep,  # 3. Step in loop
                    index=j,  # 4. Index (Batch ID)
                    lr=current_lr,
                    losses_dict=combined_accuracy_dict,  # 包含GAN和base模型的准确性
                    predictions=pred_dict,
                    targets=target_dict,
                    # 传入新东西
                    metadata=metadata,
                    model=model
                )
                allacc.append(accuracy)
                # 打印时同时显示GAN和base模型的准确性
                print(f"Window [{j + 1}/{num_windows}], GAN Accuracy: {gan_accuracy:.4f}, Base Accuracy: {accuracy:.4f}")
                if random.random() > 0.95:
                    pwindow_data = window_data
                    pwindow_labels = window_labels
                    ppred_labels = pred_labels
                    ppred_window_data = clean_ecg
                    ptarget_clean_data = target_clean_data
                    paccuracy = accuracy_dict["acc_precision"]
            allacc = np.array(allacc)
            finalacc.append(np.mean(allacc))
            if (teststep + 1) % 10 == 0:
                print(f"test准确率，step_{teststep}/{len(test_loader)}:{np.mean(allacc)}")
        finalacc = np.array(finalacc)
        print(f"alltest准确率:{np.mean(finalacc)}")
        
        # 打印按分数记录的指标
        print("\n=== 按数据分数记录的指标 ===")
        for score, metrics_list in score_metrics.items():
            if metrics_list:
                avg_precision = np.mean([m['precision'] for m in metrics_list])
                avg_recall = np.mean([m['recall'] for m in metrics_list])
                avg_f1 = np.mean([m['f1'] for m in metrics_list])
                avg_gan_precision = np.mean([m['gan_precision'] for m in metrics_list])
                avg_gan_recall = np.mean([m['gan_recall'] for m in metrics_list])
                avg_gan_f1 = np.mean([m['gan_f1'] for m in metrics_list])
                print(f"分数 {score}: Base - 准确率={avg_precision:.4f}, 召回率={avg_recall:.4f}, F1={avg_f1:.4f}")
                print(f"         GAN - 准确率={avg_gan_precision:.4f}, 召回率={avg_gan_recall:.4f}, F1={avg_gan_f1:.4f}")
    
    if pwindow_data is None:
        pwindow_data = window_data
        pwindow_labels = window_labels
        ppred_labels = pred_labels
        ppred_window_data = clean_ecg
        ptarget_clean_data = target_clean_data
        paccuracy = accuracy_dict["acc_precision"]
    # visualize_and_save(pwindow_data.cpu().numpy(), pwindow_labels.cpu().numpy(), ppred_labels, epoch,step, img_dir)
    gan_visualize_and_save(pwindow_data.cpu().numpy(), pwindow_labels.cpu().numpy(), ptarget_clean_data.cpu().numpy(),
                           ppred_labels, ppred_window_data.cpu().numpy(), epoch, step, img_dir, paccuracy)
    
    return all_preds_dict, labels

# 训练函数
def train_model(model, ganmodel, device, train_loader, test_loader, criterion, logger, num_epochs=10,
                start_epoch=start_epoch,
                update_every=5, test_every=20, img_dir='img', model_dir='models', prefix="", test_model_func=None):
    optimizer, scheduler = get_optimizer_and_scheduler(
        model,
        session_epochs=num_epochs, lr=1e-4,  # 使用较小的学习率进行微调
    )

    model.train()  # 设置为训练模式
    mining_criterion_bce = nn.BCEWithLogitsLoss(reduction='none')
    step = 0
    # 获取训练的batch size
    batch_size = train_loader.batch_size
    ganmodel.eval()
    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        epoch_loss = 0
        for i, (data, labels, paths, start_indices, scores) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}"):
            # print("new window")
            model.train()
            ganmodel.eval()
            batch_loss = 0
            data = data.unsqueeze(1)  # 变成 (batch_size, 1, seq_len)
            num_windows = (data.size(2) - (window_size - window_stride)) // window_stride  # 滑动窗口的数量
            for j in range(num_windows):

                start = j * window_stride
                end = start + window_size  # 每个滑动窗口覆盖7680个数据点
                window_data = data[:, :, start:end]
                window_labels = labels[:, start:end]
                target_clean_data = ECGSynthesizer(window_labels.to(device))
                target_clean_data = target_clean_data.to(device)
                window_data = window_data.to(device)
                wavelet_feats_np = ECGFeatureExtractor.extract_wavelet_features(window_data.cpu().detach().numpy())
                wavelet_tensor = torch.from_numpy(wavelet_feats_np).float().to(device)  # (2, L)
                window_labels = agument.expanded_probility_label(window_labels.cpu().numpy())

                gan_input_data = torch.cat([
                    window_data,  # Ch 0
                    wavelet_tensor  # Ch 3, 4
                ], dim=1)
                with torch.no_grad():
                    pred_clean_ecg, pred_r_peak = ganmodel(gan_input_data)
                pred_clean_ecg = agument.apply_clean_augmentation(pred_clean_ecg)
                wavelet_feats_np = ECGFeatureExtractor.extract_wavelet_features(pred_clean_ecg.cpu().detach().numpy())
                wavelet_tensor = torch.from_numpy(wavelet_feats_np).float().to(device)  # (2, L)
                input_data = torch.cat([
                    pred_clean_ecg,  # Ch 0
                    wavelet_tensor,  # Ch 3, 4
                    pred_r_peak
                ], dim=1)
                target, weight_map = generate_weight_from_labels(window_labels, is_soft_label=True, to_tensor=True)
                weight_map = weight_map.to(device)
                input_data = input_data.to(device)
                # 前向传播
                optimizer.zero_grad()
                window_labels = normalize_input_shape(window_labels, device=device)
                pred_r_peak = model(input_data)
                total_loss = criterion(pred_r_peak, window_labels.float(), weight_map.float())
                # loss = criterion(pred_r_peak, window_labels)  # 展平
                total_loss.backward()
                batch_loss += total_loss.item()
                optimizer.step()
                optimizer.zero_grad()
                # 每隔 update_every 次窗口，进行一次梯度更新
                if (j + 1) % update_every == 0 or (j + 1) == num_windows:
                    accuracy_dict = calc_ecg_accuracy_detailed(pred_r_peak, window_labels)
                    accuracy = accuracy_dict['acc_precision']
                    print(f"{datetime.now()}:Epoch [{epoch + 1}/{num_epochs}], "
                          f"Step [{i + 1}/{len(train_loader)}], "
                          f"Window [{j + 1}/{num_windows}], "
                          f"Loss: {total_loss.item():.4f}, "
                          f"Accuracy: {accuracy_dict['acc_precision']:.4f},"
                          f"Acc_recall: {accuracy_dict['acc_recall']:.4f},"
                          f"Acc_f1: {accuracy_dict['acc_f1']:.4f}")
                    pred_dict = {
                        'clean_ecg': target_clean_data,
                        'pred_r': pred_r_peak,
                    }

                    # 包装 Targets
                    target_dict = {
                        'clean_ecg': pred_clean_ecg,  # 无论是合成还是原始
                        'pred_r': window_labels.float()
                    }
                    current_lr = optimizer.param_groups[0]['lr']
                    # 一键记录
                    with torch.no_grad():
                        # 我们用 Model B 的分割误差来衡量样本难易程度
                        # BCE per pixel: (B, 1, L)
                        raw_bce = mining_criterion_bce(pred_r_peak, window_labels)
                        # 加上 weight map (关注困难区域)
                        weighted_bce = raw_bce * weight_map
                        # 对每个样本求平均 -> (B,)
                        sample_losses = weighted_bce.mean(dim=(1, 2))
                    metadata = (paths, start_indices, sample_losses)
                    metrics = logger.log_step(
                        mode='train',
                        epoch=epoch,  # 1. Epoch
                        global_step=step,  # 2. Global Step
                        step=i,  # 3. Step in loop
                        index=j,  # 4. Index (Batch ID)
                        lr=current_lr,
                        losses_dict=accuracy_dict,  # 也就是你 loss 函数返回的 {l1:..., stft:...}
                        predictions=pred_dict,
                        targets=target_dict,
                        # 传入新东西
                        metadata=metadata,
                        model=model
                    )
            # swanlab.log({"batch_loss": batch_loss})
            epoch_loss += batch_loss
            print(f"{datetime.now()}:Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Batch Loss: {batch_loss/num_windows:.4f}")
            if (i + 1) % test_every == 0:
                print(f"Performing test after {i + 1} steps")
                test_model_func(model, ganmodel, device, test_loader, logger, epoch, step, img_dir, batch_size=batch_size)  # 调用test函数进行测试

            step += 1
        model.save_weights(model_dir, epoch, step, prefix=prefix)
        # 输出每个 epoch 的平均损失
        print(f'Test {datetime.now()}:Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}')
        # 每个 epoch 后更新学习率
        # scheduler.step(epoch_loss)
        scheduler.step()
        # 打印当前 LR 看看变化
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch} finished. Current LR: {current_lr:.6f}")
        logger.flush(mode="train")
    logger.close()

# 微调函数
def fine_tune_model(model_type, model_path, train_loader, test_loader):
    """微调预训练模型"""
    print(f"\n=== 微调 {model_type} 模型 ===")
    
    # 初始化模型
    ganmodel = ECGDetectionSystem(input_length=15360,
                               in_channels=3, out_channels=2, model="tra")
    ganmodel.load_weights(ganmodel_dir)
    ganmodel.to(device)
    
    model = ECGDetectionSystem(input_length=15360,
                               in_channels=4, out_channels=1, model="tra")
    criterion = AggressiveSegmentationLoss().to(device)
    
    # 加载预训练模型
    try:
        model.load_weights(model_path)
        print(f"成功加载模型: {model_path}")
    except FileNotFoundError:
        print(f"错误: 模型文件未找到: {model_path}")
        return
    
    # 创建日志记录器
    logger = TrainingLogger(
        log_dir=model_dir,
        train_name=f"correct_ganbase_{model_type}",
        heavy_mode=True
    )
    
    model.to(device)
    
    # 开始微调
    train_model(model, ganmodel, device, train_loader, test_loader, criterion, logger, num_epochs=20, start_epoch=0,
                update_every=50, test_every=20, img_dir=img_dir, model_dir=model_dir, 
                prefix=f'correct_ganbase_{model_type}', test_model_func=test_model)

# 主程序
if __name__ == '__main__':
    # 初始化合成器和数据增强
    ECGSynthesizer = AdaptiveECGSynthesizer(sampling_rate=512, device=device)
    agument = ECGDataAugmentation(augment=True, expanded=True)
    
    # 准备数据 - 90%训练，10%测试
    print("=== 准备数据 (90%训练, 10%测试) ===")
    
    # 使用excellent_data.hdf文件
    hdf_path = "/root/PycharmProjects/heartfenxi/ablistimunet/excellent_data.hdf"
    traindataset = ExcellentDataDataset(hdf_path, window_size=15360, stride=7680, test_size=0.1, random_seed=42)
    traindataset.setTrainStatus(train=True)
    testdataset = ExcellentDataDataset(hdf_path, window_size=15360, stride=7680, test_size=0.1, random_seed=42)
    testdataset.setTrainStatus(train=False)
    
    train_loader = ECGDataLoader(traindataset, batch_size=4, shuffle=True, test=False, num_workers=4)
    test_loader = ECGDataLoader(testdataset, batch_size=1, shuffle=True, test=True)
    print(f"Number of training samples: {len(train_loader)}", f"Number of test samples: {len(test_loader)}")
    
    # 定义最新模型路径
    latest_model_path = os.path.join("./modelAll/modeldirganbase1/", 'model_9_1810.pth')
    
    # 微调最新模型
    if os.path.exists(latest_model_path):
        fine_tune_model('ganbase', latest_model_path, train_loader, test_loader)
    else:
        print(f"模型文件不存在: {latest_model_path}")
