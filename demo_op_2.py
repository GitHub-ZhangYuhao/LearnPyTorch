﻿import torch
a = torch.rand(2,2)
a *= 10
print(a)

print(torch.floor(a)) #向下取整
print(torch.ceil(a))  #向上取整
print(torch.round(a)) #四舍五入
print(torch.trunc(a)) #取整数部分
print(torch.frac(a)) #取小数部分
print(a % 2) #取a/2的余数