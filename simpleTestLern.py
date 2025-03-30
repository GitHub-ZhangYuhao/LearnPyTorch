import torch

'''
自动微分
'''
x = torch.tensor(3, dtype=torch.float16, requires_grad=True)
y = x**2        #y=x^2
y.backward()    #计算dy/dx
print(x.grad)   #输出x的梯度 (y=x^2 的导数 是y=2x, 所以输出6)
#梯度清零 （训练中必须）
x.grad.zero_()