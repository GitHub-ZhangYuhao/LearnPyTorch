import torch

a = torch.rand(2,2)

print(a)
print(torch.mean(a)) # 计算a平均值
print(torch.sum(a)) # 计算a所有元素的和
print(torch.prod(a)) # 计算a所有元素的积

print(torch.mean(a, dim=0)) # 计算a每一列的平均值
print(torch.sum(a, dim=0)) # 计算a每一列的和
print(torch.prod(a, dim=0)) # 计算a每一列的积

print(torch.argmax(a, dim=0)) # 计算a每一列的最大值的 索引
print(torch.argmin(a, dim=0)) # 计算a每一列的最小值的 索引

print(torch.std(a)) # 计算a所有元素的标准差
print(torch.var(a)) # 计算a所有元素的方差

print(torch.median(a)) # 计算a所有元素的中位数
print(torch.mode(a)) # 计算a所有元素的众数

a = torch.rand(2,2) * 10
print(a)
# 计算a所有元素的直方图,分为 bins 条,最大值和最小值为当前tensor的最大值和最小值
print(torch.histc(a, bins=10, min=0, max=0 ))

a = torch.randint(0,10, [10]) # 生成一个0~10的随机整数, 共10个
print(a)
#频次统计，只能支持一维tensor，统计每个定元素出现的次数
print(torch.bincount(a)) # 统计a中每个元素出现的次数
#可以用来统计某一类别样本的个数