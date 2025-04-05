import torch

'''
自动微分
'''
x = torch.tensor(3, dtype=torch.float16, requires_grad=True)
y = x**2        #y=x^2
y.backward()    #计算dy/dx
#print(x.grad)   #输出x的梯度 (y=x^2 的导数 是y=2x, 所以输出6)
#梯度清零 （训练中必须）
x.grad.zero_()



'''
Python NNE Model Create (UE5.5 NNE )
https://dev.epicgames.com/community/learning/tutorials/7dr8/unreal-engine-nne-neural-post-processing#modelassetcreation
'''
# import torch.nn as nn
# class SobelFilter(torch.nn.Module):
#
#     def __init__(self):
#         super(SobelFilter, self).__init__()
#         self.hFilter = torch.tensor([[[[1, 0, -1], [2, 0, -2], [1, 0, -1]]] * 3] * 3, dtype=torch.float)
#         self.vFilter = torch.tensor([[[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]] * 3] * 3, dtype=torch.float)
#
#     def forward(self, x):
#         x = torch.nn.functional.pad(x, (1, 1, 1, 1), mode='replicate')
#         h = torch.nn.functional.conv2d(x, self.hFilter)
#         v = torch.nn.functional.conv2d(x, self.vFilter)
#         return torch.sqrt(h * h + v * v)
#
# if __name__=="__main__":
#     model = SobelFilter()
#     input = torch.randn(1, 3, 256, 256)
#     onnx = torch.onnx.export(model, (input,), 'sobel.onnx',
#                              input_names=['input'], output_names=['output'], opset_version=9,
#                              dynamic_axes={'input': {2: 'height', 3: 'width'}, 'output' : {2: 'height', 3: 'width'}})
'''
End
'''



'''
二维卷积演示代码
分别使用了： 
1. nn.functional.conv2d 和 
2. nn.Conv2d 
两种方式进行二维卷积操作 示例
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

in_channels = 1
out_channels = 1
kernel_size = 3
bias = True
batch_size = 1
input_size = [batch_size, in_channels, 4, 4]
conv_layer = nn.Conv2d(in_channels, out_channels,kernel_size, bias=bias)
#print("Conv_Weight: ", conv_layer.weight) # 1*1*3*3 = out_channels * in_channels * height * width
input_feature_map = torch.randn(input_size)
#print("\ninput_feature_map:\n ", input_feature_map)
output_feature_map = conv_layer(input_feature_map)
#print("\noutput_feature_map\n: ", output_feature_map)

# 使用 nn.functional.conv2d 进行卷积操作
# 卷积核的Tensor的Shape应该为：输出通道数 * 输入通道数 * 卷积核高度 * 卷积核宽度
kernel_weight = torch.tensor([ [[[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]]] ],
                             dtype=torch.float16)
print("kernel_weight_shape: \n", kernel_weight.shape)
input_feature_map = torch.rand((batch_size,in_channels,4,4), dtype=torch.float16)
output_feature_map1 = nn.functional.conv2d(input_feature_map, kernel_weight)
print("\noutput_feature_map1:\n", output_feature_map1)
