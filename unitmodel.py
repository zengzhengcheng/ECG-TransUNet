import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ablistimunet.model import ECGDetectionSystem
import torch
# 获取当前文件所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
basemodelpath=os.path.join(current_dir, "usemodel", "trabase.pth")
ganmodelpath=os.path.join(current_dir, "usemodel", "tragan.pth")
ganbasemodelpath=os.path.join(current_dir, "usemodel", "traganbase.pth")
model_ganbase=ECGDetectionSystem(input_length=15360,
                               in_channels=4, out_channels=1, model="tra")
model_gan=ECGDetectionSystem(input_length=15360,
                               in_channels=3, out_channels=2, model="tra")
model_base=ECGDetectionSystem(input_length=15360,
                               in_channels=3, out_channels=1, model="tra")
model_ganbase.load_weights(ganbasemodelpath)
model_ganbase.eval()
model_gan.load_weights(ganmodelpath)
model_gan.eval()
model_base.load_weights(basemodelpath)
model_base.eval()
dynamic_axes = {
    'input': {0: 'batch_size'},
    'ECG': {0: 'batch_size'},
    'pred_r': {0: 'batch_size'}
}
dummy_input = torch.randn(1, 3, 15360)  # 示例输入 (batch=1, channel=1, length=15360)

# 确保输出目录存在
os.makedirs(os.path.join(current_dir, "outmodel"), exist_ok=True)

torch.onnx.export(
    model_gan,
    dummy_input,
    os.path.join(current_dir, "outmodel", "tragan_model.onnx"),
    input_names=["input"],
    output_names=["ECG", "pred_r"],  # 对应模型输出的两个张量
    dynamic_axes=dynamic_axes,
    opset_version=14,  # 推荐较高版本确保兼容性[2](@ref)
    do_constant_folding=True  # 启用常量折叠优化
)
dynamic_axes = {
    'input': {0: 'batch_size'},
    'pred_r': {0: 'batch_size'}
}
dummy_input = torch.randn(1,3, 15360)  # 示例输入 (batch=1, channel=1, length=15360)
torch.onnx.export(
    model_base,
    dummy_input,
    os.path.join(current_dir, "outmodel", "trabase_model.onnx"),
    input_names=["input"],
    output_names=["pred_r"],  # 对应模型输出的两个张量
    dynamic_axes=dynamic_axes,
    opset_version=14,  # 推荐较高版本确保兼容性[2](@ref)
    do_constant_folding=True  # 启用常量折叠优化
)
dynamic_axes = {
    'input': {0: 'batch_size'},
    'pred_r': {0: 'batch_size'}
}
dummy_input = torch.randn(1, 4, 15360)  # 示例输入 (batch=1, channel=1, length=15360)
torch.onnx.export(
    model_ganbase,
    dummy_input,
    os.path.join(current_dir, "outmodel", "traganbase_model.onnx"),
    input_names=["input"],
    output_names=["pred_r"],  # 对应模型输出的两个张量
    dynamic_axes=dynamic_axes,
    opset_version=14,  # 推荐较高版本确保兼容性[2](@ref)
    do_constant_folding=True  # 启用常量折叠优化
)