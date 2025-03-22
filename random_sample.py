import torch

torch.manual_seed(1) #在定义好随机种子后，每次生成的随机数都是一样的
mean = torch.tensor([8],dtype=torch.float32)#torch.rand(1,2)
std  = torch.tensor([0.01])#torch.rand(1,2)
print(mean)
print(std)
print(torch.normal(mean=mean , std=std)) # 生成一个正态分布的随机数
'''
tensor([8.])
tensor([0.0100])
tensor([8.0066])
生成的结果 = 均值 +/- 标准差*随机值
'''