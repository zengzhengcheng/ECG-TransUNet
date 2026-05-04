import os.path
import random

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ablistimunet.ECGDataset import ECGDataset, ECGDataLoader,ECG6LabelDataset # 假设 ECGDataset 类已经正确实现
from ablistimunet.ECGDataAugmentation import ECGDataAugmentation
from datetime import datetime
from gongju import *
import configparser
from torch.optim.lr_scheduler import ReduceLROnPlateau,CosineAnnealingLR
from model import ECGDetectionSystem
from ablistimunet.utils import ECGFeatureExtractor
from ablistimunet.utils import TrainingLogger,MetricCalculator
from ablistimunet.losses import AdvancedSegmentationLoss
from ablistimunet.losses import AdvancedGenerativeLoss
from ablistimunet.accuracyMeasure.evaluate_accurate_data import AccurateDataEvaluator
# 模型定义
config = configparser.ConfigParser()
# 读取配置文件
config.read('config.ini',encoding="utf-8")  # 假设 config.ini 文件与 main.py 在同一目录下
machine_list=["macmachine","ubuntumachine","winmachine"]
dir_list=["common","dirgan","dirgan1","dirgan2"]

dir_name=dir_list[2]
machine=machine_list[1]
# 获取配置信息
data_dir = config.get(machine, 'data_dir')  # 从 [DEFAULT] 节获取 data_dir 选项
img_dir = config.get(dir_name, 'img_dir')
img_dir = os.path.join("imgAll",img_dir)
model_dir = config.get(dir_name, 'model_dir')
model_dir =os.path.join('modelAll',model_dir)
model_dir = os.path.join(os.getcwd(),model_dir)
dir_name=dir_list[0]
window_stride = config.getint(dir_name, 'window_stride') # 注意这里使用 getint 获取整数值
log_dir=config.get(dir_name, 'log_dir')
useSwanLab = config.getboolean(dir_name, 'useSwanLab')
totallength=30
baselength=0
predictlength=totallength-baselength
start_epoch=0
epochs=20
sample_rate=512
window_size = totallength*sample_rate
# 定义要加载的模型文件名 (根据您的实际文件名修改)
model_filename = 'model_9_1810.pth'
model_path = os.path.join(model_dir, model_filename)
torch.manual_seed(42)
# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available()
                      else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(device)

def test_model(model, device,test_loader,epoch,step,img_dir, accurate_data_dir=None, batch_size=8):
    model.eval()  # 设置为评估模式
    all_preds_dict = {}  # 用字典保存每个数据点的预测标签
    pwindow_data=None
    mining_criterion_bce = nn.BCEWithLogitsLoss(reduction='none')
    with torch.no_grad():  # 测试时不计算梯度
        finalacc=[]
        for teststep, (data, labels, paths, start_indices) in enumerate(test_loader):
            allacc = []
            data = data.unsqueeze(1)  # 变成 (batch_size, 1, seq_len)
            num_windows = (data.size(2) - (window_size-window_stride)) // window_stride
            for j in range(num_windows):
                start = j * window_stride
                end = start + window_size  # 每个滑动窗口覆盖7680个数据点
                window_data = data[:, :, start:end]
                window_data = window_data.to(device)
                wavelet_tensor = ECGFeatureExtractor.extract_wavelet_features(window_data,tensor=True,device=device)
                input_data = torch.cat([
                    window_data,  # Ch 0
                    wavelet_tensor  # Ch 3, 4
                ], dim=1)
                window_labels = labels[:, start:end]
                window_labels = window_labels.to(device).float()
                target_clean_data = ECGSynthesizer(window_labels.to(device))
                target_clean_data = target_clean_data.to(device)
                window_labels=agument.expanded_probility_label(window_labels,tensor=True,device=device)
                # 前向传播获取预测
                outputs = model(input_data )
                clean_ecg,pred_r_peak = outputs  # 后处理输出
                probabilities = torch.sigmoid(pred_r_peak)

                # 设定阈值（0.5）得到二分类标签
                threshold = 0.5
                pre_labels = (probabilities >= threshold).float()
                pre_labels=pre_labels.squeeze()
                # 获取预测标签和概率
                pred_labels = pre_labels.cpu().numpy()  # 预测标签
                accuracy_dict = calc_ecg_accuracy_detailed(pred_r_peak,window_labels)
                allacc.append(accuracy_dict["acc_precision"])
                pred_dict = {
                    'clean_ecg': target_clean_data,
                    'pred_r': pred_r_peak,
                }

                # 包装 Targets
                target_dict = {
                    'clean_ecg': clean_ecg,  # 无论是合成还是原始
                    'pred_r': window_labels
                }
                current_lr = 0
                # 一键记录
                with torch.no_grad():
                    # 我们用 Model B 的分割误差来衡量样本难易程度
                    # BCE per pixel: (B, 1, L)
                    pred_r_peak = normalize_input_shape(pred_r_peak,device=device)
                    window_labels = normalize_input_shape(window_labels,device=device)
                    raw_bce = mining_criterion_bce(pred_r_peak, window_labels)
                    # 加上 weight map (关注困难区域) - 与训练过程保持一致
                    target, weight_map = generate_weight_from_labels(window_labels, is_soft_label=True, to_tensor=True)
                    weight_map = weight_map.to(device)
                    weighted_bce = raw_bce * weight_map
                    # 对每个样本求平均 -> (B,)
                    sample_losses = weighted_bce.mean(dim=(1, 2))
                metadata = (paths, start_indices, sample_losses)
                metrics = logger.log_step(
                    mode='test',
                    epoch=epoch,  # 1. Epoch
                    global_step=step,  # 2. Global Step
                    step=teststep,  # 3. Step in loop
                    index=j,  # 4. Index (Batch ID)
                    lr=current_lr,
                    losses_dict=accuracy_dict,  # 也就是你 loss 函数返回的 {l1:..., stft:...}
                    predictions=pred_dict,
                    targets=target_dict,
                    # 传入新东西
                    metadata=metadata,
                    model=model
                )
                if(random.random()>0.95):
                    pwindow_data = window_data
                    pwindow_labels = window_labels
                    target_clean_data = agument._generate_adaptive_median_target(window_data, window_labels,device=device)
                    if target_clean_data.abs().max() < 0.1:
                        target_clean_data = ECGSynthesizer(window_labels.to(device))
                    else:
                        target_clean_data = target_clean_data
                    ptarget_clean_data=target_clean_data
                    ppred_labels = pred_labels
                    ppred_window_data = clean_ecg
                    paccuracy=accuracy_dict["acc_precision"]
            allacc=np.array(allacc)
            finalacc.append(np.mean(allacc))
            if((teststep+1)%10==0):
                print(f"test准确率，step_{teststep}/{len(test_loader)}:{np.mean(allacc)}")
            if(random.random()>0.96):
                break
        finalacc=np.array(finalacc)
        print(f"alltest准确率:{np.mean(finalacc)}")
    if(pwindow_data is None):
        pwindow_data = window_data
        pwindow_labels = window_labels
        ppred_labels = pred_labels
        ppred_window_data = clean_ecg
        ptarget_clean_data = target_clean_data
        paccuracy = accuracy_dict["acc_precision"]
    # visualize_and_save(pwindow_data.cpu().numpy(), pwindow_labels.cpu().numpy(), ppred_labels, epoch,step, img_dir)
    gan_visualize_and_save(pwindow_data.cpu().numpy(), pwindow_labels.cpu().numpy(),ptarget_clean_data.cpu().numpy(),
                           ppred_labels, ppred_window_data.cpu().numpy(), epoch, step, img_dir,paccuracy)
    
    # 评估准确数据
    if accurate_data_dir:
        print("Evaluating accurate data...")
        log_dir = model_dir
        evaluator = AccurateDataEvaluator(log_dir)
        evaluator.evaluate(model, device, accurate_data_dir, batch_size=batch_size)
    
    return all_preds_dict,labels

# 训练函数
def train_model(model, device,train_loader, criterion, num_epochs=10,
                start_epoch=start_epoch,
                update_every=5, test_every=20, img_dir='img',model_dir='models',prefix="", accurate_data_dir=None):
    optimizer, scheduler = get_optimizer_and_scheduler(
        model,
        session_epochs=num_epochs, lr=1e-3,
    )
    model.train()  # 设置为训练模式
    mining_criterion_bce = nn.BCEWithLogitsLoss(reduction='none')

    step = 0
    # 获取训练的batch size
    batch_size = train_loader.batch_size
    for epoch in range(start_epoch,start_epoch+num_epochs):
        model.train()
        epoch_loss = 0
        for i, (data, labels, paths, start_indices) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}"):
            # print("new window")
            model.train()
            batch_loss = 0
            data = data.unsqueeze(1)  # 变成 (batch_size, 1, seq_len)
            num_windows = (data.size(2) - (window_size-window_stride)) // window_stride  # 滑动窗口的数量
            for j in range(num_windows):
                start = j * window_stride
                end = start + window_size  # 每个滑动窗口覆盖7680个数据点
                window_data = data[:, :, start:end]
                window_labels = labels[:, start:end]
                if (random.random() > 0.5):
                    with torch.no_grad():
                        target_clean_data = ECGSynthesizer(window_labels.to(device))
                        target_clean_data = target_clean_data.to(device)
                        loss_weight_scale = 2.0  # 转为强约束
                else:
                    target_clean_data=agument._generate_adaptive_median_target(window_data,window_labels)
                    if target_clean_data.abs().max() < 0.1:
                        target_clean_data = ECGSynthesizer(window_labels.to(device))
                        loss_weight_scale = 2.0  # 转为强约束
                    else:
                        target_clean_data = target_clean_data
                        loss_weight_scale = 1.0
                    target_clean_data=target_clean_data.to(device)
                if(random.random() > 0.5):
                    window_data=agument.apply_augmentation(window_data)
                window_labels=agument.expanded_probility_label(window_labels.cpu().numpy())
                if random.random() > 0.5:
                    window_data, window_labels = agument.apply_chaos(window_data, window_labels)
                #获得小波数据 (基于增强后的ECG信号)
                wavelet_feats_np = ECGFeatureExtractor.extract_wavelet_features(window_data)
                wavelet_tensor = torch.from_numpy(wavelet_feats_np).float().to(device)  # (2, L)
                input_noisy_data = window_data.to(device)
                if(not torch.is_tensor(window_labels)):
                    window_labels=torch.from_numpy(window_labels)
                window_labels= window_labels.to(device).float()  # 目标标签是后10秒标签
                window_labels=normalize_input_shape(window_labels)
                window_labels=window_labels.to(device)
                # 前向传播
                optimizer.zero_grad()
                input_data= torch.cat([
                    input_noisy_data,  # Ch 0
                    wavelet_tensor  # Ch 3, 4
                ], dim=1)
                pred_clean_ecg, pred_r_peak = model(input_data)
                target,weight_map = generate_weight_from_labels(window_labels,is_soft_label=True,to_tensor=True)
                weight_map=weight_map.to(device)
                # 4. 计算损失
                total_loss, loss_dict  = criterion(
                    (pred_clean_ecg, pred_r_peak),
                    (target_clean_data, window_labels,weight_map),loss_weight_scale
                )
                # loss = criterion(pred_r_peak, window_labels)  # 展平
                total_loss.backward()
                batch_loss += total_loss.item()
                optimizer.step()
                optimizer.zero_grad()
                # 每隔 update_every 次窗口，进行一次梯度更新
                if (j + 1) % update_every == 0 or (j + 1) == num_windows:
                    accuracy_dict=calc_ecg_accuracy_detailed(pred_r_peak,window_labels)
                    print(f"{datetime.now()}:Epoch [{epoch + 1}/{num_epochs}], "
                          f"Step [{i + 1}/{len(train_loader)}], "
                          f"Window [{j + 1}/{num_windows}], "
                          f"Loss: {total_loss.item():.4f}, "
                          f"Accuracy: {accuracy_dict['acc_precision']:.4f},"
                          f"Acc_recall: {accuracy_dict['acc_recall']:.4f},"
                          f"Acc_f1: {accuracy_dict['acc_f1']:.4f}")
                    print(f"L1: {loss_dict['l1']:.4f}, Grad: {loss_dict['grad']:.4f}, STFT: {loss_dict['stft']:.4f},"
                          f"peak: {loss_dict['peak']:.4f}")
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
                        losses_dict=loss_dict,  # 也就是你 loss 函数返回的 {l1:..., stft:...}
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
                test_model(model,device,test_loader,epoch,step,img_dir, accurate_data_dir, batch_size=batch_size)  # 调用test函数进行测试

            step += 1
        model.save_weights(model_dir,epoch, step,prefix=prefix)
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

# 主程序
if __name__ == '__main__':
    model = ECGDetectionSystem(input_length=15360,
                               in_channels=3, out_channels=2, model="tra")
    criterion = AdvancedGenerativeLoss().to(device)
    ECGSynthesizer = AdaptiveECGSynthesizer(sampling_rate=512, device=device)
    logger = TrainingLogger(
        log_dir=model_dir,
        train_name="GAN_baddata_10",
        heavy_mode=True  # 开关在这里控制
    )
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # scheduler = CosineAnnealingLR(optimizer, T_max=10)
    if (model_filename != ""):
        try:
            model.load_weights(model_path)
            print(f"成功加载模型: {model_filename}")
        except FileNotFoundError:
            print(f"错误: 模型文件未找到: {model_path}")
    agument = ECGDataAugmentation(augment=True, expanded=True)
    model.to(device)
    if(False):
        # 第一阶段：使用基础数据训练10个epoch
        print("\n=== 第一阶段：使用基础数据训练 ===")
        testdataset = ECGDataset(data_dir=data_dir, window_size=307200, stride=153600, test_size=0.1, wave=True)
        traindataset = ECGDataset(data_dir=data_dir, window_size=307200, stride=153600, test_size=0.1, wave=True)
        train_loader = ECGDataLoader(traindataset, batch_size=16, shuffle=True, test=False, num_workers=2)
        test_loader = ECGDataLoader(testdataset, batch_size=1, shuffle=True, test=True)
        print(f"Number of training samples: {len(train_loader)}", f"Number of test samples: {len(test_loader)}")
        # 准确数据目录
        accurate_data_dir = "/root/PycharmProjects/heartfenxi/ablistimunet/accuracyMeasure/test_data"  # 准确数据目录
        train_model(model,device, train_loader,  criterion, num_epochs=10,start_epoch=0,
                    update_every=50, test_every=20, img_dir=img_dir,model_dir=model_dir, accurate_data_dir=accurate_data_dir)
        
    # 第二阶段：使用bad数据训练10个epoch
    print("\n=== 第二阶段：使用bad数据训练 ===")
    # 创建新的日志记录器，使用不同的名称
    logger = TrainingLogger(
        log_dir=model_dir,
        train_name="GAN_baddata_10",
        heavy_mode=True  # 开关在这里控制
    )
    testdataset = ECG6LabelDataset(data_dir=data_dir,expanded=True, window_size=153600, stride=76800, test_size=0.1,good=0.01)
    traindataset = ECG6LabelDataset(data_dir=data_dir,expanded=True,window_size=153600, stride=76800, test_size=0.1,good=0)
    train_loader = ECGDataLoader(traindataset, batch_size=4, shuffle=True, test=False,num_workers=4)
    test_loader = ECGDataLoader(testdataset, batch_size=1, shuffle=True, test=True)
    print(f"Number of training samples: {len(train_loader)}", f"Number of test samples: {len(test_loader)}")
    # 准确数据目录
    accurate_data_dir = "/root/PycharmProjects/heartfenxi/ablistimunet/accuracyMeasure/test_data"  # 准确数据目录
    train_model(model, device, train_loader, criterion, num_epochs=10, start_epoch=10,
                 update_every=50, test_every=20, img_dir=img_dir, model_dir=model_dir, prefix='baddata', accurate_data_dir=accurate_data_dir)