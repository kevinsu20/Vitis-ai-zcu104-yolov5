import torch
import numpy as np

# 加载量化模型
model = torch.load('bias_corr.pth')

# 读取指定层的 Bias
layer_name = 'conv1.bias'
bias_value = model['state_dict'][layer_name].numpy()

# 打印 Bias 的形状和值
print('Bias shape:', bias_value.shape)
print('Bias value:', bias_value)
