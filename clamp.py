import torch

a = torch.rand(2, 2) * 10 #生成一个2*2的随机矩阵，每个元素的值在0到10之间
print(a)
a = a.clamp(2,5) #将a的每一个元素限制在2到5之间
print(a)