import torch

'''
三角函数
'''
a = torch.ones(2,3)
a *= torch.pi/4 #将a的每一个元素乘以pi/4 (45°)
b = torch.cos(a) #对a的每一个元素进行cos计算
print(a)
print(b)