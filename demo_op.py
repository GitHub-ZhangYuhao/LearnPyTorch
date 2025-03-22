import torch

a = torch.rand(2,3)
b = torch.rand(2,3)
print(a)
print(b)

## add
print("-------add-------")
print(a+b)
print(a.add(b))
print(torch.add(a,b))
print(a.add_(b))
print(a)

## sub
print("-------sub-------")
print(a-b)
print(a.sub(b))
print(torch.sub(a,b))
print(a.sub_(b))
print(a)

## mul
print("-------mul-------")
print(a*b)
print(torch.mul(a,b))
print(a.mul(b))
print(a.mul_(b))
print(a)

## div
print("-------div-------")
print(a/b)
print(torch.div(a,b))
print(a.div(b))
print(a.div_(b))
print(a)

## matmul
print("-------matmul-------")
a = torch.ones(2,1)
b = torch.ones(1,2)
print(a @ b)
print(a.matmul(b))
print(torch.matmul(a,b))
print(torch.mm(a,b))
print(a.mm(b))

## 高维 tensor
print("-------高维 tensor-------")
#高维度矩阵的运算后两位需要保证是能够进行矩阵运算的，如：a*b 和 b*c 矩阵，前面的阶数需要一样
a = torch.ones(1,2,3,4)
b = torch.ones(1,2,4,3)
print(a.matmul(b))
print(a.matmul(b).shape)

## pow指数运算 a^3
print("------- pow -------")
a = torch.tensor([1,2])
print(torch.pow(a,3))
print(a.pow(3))
print(a**3)
print(a.pow_(3))

## exp 自然常熟指数运算 e^a
print("------- exp -------")
a = torch.tensor([1,2], dtype=torch.float32)
print(torch.exp(a))
print(torch.exp_(a)) #需要浮点类型
print(a.exp())
print(a.exp_())

## log 对数运算
print("------- log -------")
a = torch.tensor([10,torch.e**2], dtype=torch.float32)
print(torch.log(a)) #e为底
print(torch.log_(a)) #会修改a的值
print(a.log())
print(a.log_())#会修改a的值

## sqrt 开根
print("------- sqrt -------")
a = torch.tensor([1,4], dtype=torch.float32)
print(torch.sqrt(a))
print(torch.sqrt_(a))
print(a.sqrt())
print(a.sqrt_())