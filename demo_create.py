import numpy as np
import torch

# 使用 torch.tensor 创建张量
a = torch.Tensor( [[1, 2],[3,4]] ) #通过值创建张量
# print(a)
# print(a.type())

a = torch.Tensor(2,3) #通过张量创建，值未初始化
# print(a)
# print(a.type())
# print(a.size())

''' 几种特殊的 tensor '''
a = torch.ones(2,2,2) #创建一个2*2*2的张量，值全为1
a = torch.zeros( 2,2,2) #创建一个2*2*2的张量，值全为0
a = torch.eye(2,2) #创建一个2*2的张量，对角线为1，其余为0
b = torch.zeros_like(a) #创建一个和a相同大小的张量，值全为0
b = torch.ones_like(a) #创建一个和a相同大小的张量，值全为1

''' 随机 '''
a = torch.rand(2, 2) #创建一个2*2的张量，值为0~1的随机数
a = torch.normal(mean=10.0 , std=torch.rand(5)) #创建一个5维的张量，值为均值为10，标准差为随机数的正态分布
a = torch.normal(mean=torch.rand(5), std=torch.rand(5)) #创建一个5维的张量，值为均值为随机数，标准差为随机数的正态分布
a = torch.Tensor(2,2).uniform_(-1,1) #创建一个2*2的张量，值为-1~1的随机数

''' 序列 '''
a = torch.arange(1, 10, 1) #创建一个1~10，步长为 1 的序列
# tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
a = torch.linspace(2, 10, 4) #创建一个2~10等间隔（等差数列），元素个数为 4 的序列
# tensor([ 2.0000,  4.6667,  7.3333, 10.0000]) 间隔都为 2.6667
a = torch.randperm(10) #创建一个0~10的随机序列，元素个数为 10, 值不包含10
# tensor([3, 4, 8, 1, 2, 5, 0, 9, 6, 7])
print(a)
print(a.type())



########### numpy ###########
import numpy as np
'''numpy 和 tensor 的结构在使用上有很大的相似性'''
a = np.array([[1,2],[3,4]]) #使用numpy定义2*2的数据结构
print(a)