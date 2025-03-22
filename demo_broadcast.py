import torch

#广播机制可以用在加减乘除的数学运算
a = torch.rand(2,3)
b = torch.rand(3)
c = a+b
print(a)
print(b)
print(c)
print(c.shape)
# a, 2*3
# b, 1*3
# c, 2*3

a = torch.rand(2, 1, 1, 2)
b = torch.rand(4, 3, 2)
c = a-b
print(a)
print(b)
print(c)
print(c.shape)
#c.Shape = (2, 4, 3, 2)