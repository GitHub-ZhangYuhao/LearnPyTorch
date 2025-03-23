import torch

a = torch.tensor([3., 4.])#torch.rand(1,2)
b = torch.rand(1,2)
print(a,"\n", b)
print(torch.dist(a, b, p=1)) # 计算a和b的L1范数
print(torch.dist(a, b, p=2)) # 计算a和b的L2范数
print(torch.dist(a, b, p=3)) # 计算a和b的L3范数

print(torch.norm(a)) #|a|( sqrt(x^2 + y^2) ) 计算a的模长 # 等价于 torch.dist(a, torch.zeros_like(a), p=2)
print(torch.norm(a, p=1)) #计算a的L1范数 (x + y)
print(torch.norm(a, p=3)) #计算a的L3范数 pow((x^3 + y^3), 1/3))
