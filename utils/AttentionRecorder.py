class AttentionRecorder:
    """
    非侵入式注意力记录器。
    原理：利用 PyTorch 的 forward hook 机制，在不修改模型代码的情况下获取中间层输出。
    """

    def __init__(self, model):
        self.hooks = []
        self.attention_maps = {}  # 存储 {层名: attention_tensor}

        # 自动遍历模型，找到所有 MultiheadAttention 层并注册
        for name, module in model.named_modules():
            # 兼容 PyTorch 原生 Transformer 和自定义 Attention
            if "MultiheadAttention" in module.__class__.__name__ or "Attention" in name:
                # 注册 hook
                hook = module.register_forward_hook(self._get_hook(name))
                self.hooks.append(hook)

    def _get_hook(self, name):
        def hook(module, input, output):
            # output 的格式取决于具体实现。
            # PyTorch MultiheadAttention 的 output 是 (attn_output, attn_weights) 如果 need_weights=True
            # 或者是仅 attn_output。你需要确保模型定义里的 forward 返回了 weights，
            # 或者 output 本身就是 weights (取决于自定义实现)。

            # 假设 output 是 (output_tensor, weights)
            if isinstance(output, tuple) and len(output) > 1:
                # 只取第一个样本的注意力图，节省内存，因为我们只用来画图分析
                # weights shape 通常是 (Batch, Seq_Len, Seq_Len)
                self.attention_maps[name] = output[1][0].detach().cpu()
            elif isinstance(output, torch.Tensor):
                # 某些实现可能只返回 tensor，这需要结合具体模型结构调整
                pass

        return hook

    def get_and_clear(self):
        """获取当前步的注意力图并清空，防止显存泄漏"""
        data = self.attention_maps.copy()
        self.attention_maps.clear()
        return data

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()