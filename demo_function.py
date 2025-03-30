from tokenize import Funny

import torch


'''
继承Function类，实现自己的函数
重写Function类的forward和backward方法
forward方法用于计算函数的输出
backward方法用于计算函数的梯度
'''
class line(torch.autograd.Function):
    @staticmethod
    def forward(ctx , w, x, b): # ctx是一个上下文对象，用于存储反向传播时需要用到的信息
        # y = w*x + b
        ctx.save_for_backward(w, x, b) # 保存反向传播时需要用到的信息,保存w,x,b
        return w*x + b # 返回y

    '''
    反向传播的目的是计算损失函数关于每个输入参数（w,x,b）的梯度
    '''
    @staticmethod
    def backward(ctx, grad_out): #需要传入ctx(上下文管理器)，和上一级的梯度 ，
        w, x, b = ctx.saved_tensors # 取出保存的信息

        # 计算梯度
        grad_w = grad_out * x # w的导数
        grad_x = grad_out * w # X的导数
        grad_b = grad_out # B的导数
        return grad_w, grad_x, grad_b # 返回梯度

# W = torch.rand(2, 2, requires_grad=True) # 生成一个2*2的随机矩阵，每个元素的值在0到1之间
# X = torch.rand(2, 2, requires_grad=True) # 生成一个2*2的随机矩阵，每个元素的值在0到1之间
# B = torch.rand(2, 2, requires_grad=True) # 生成一个2*2的随机矩阵，每个元素的值在0到1之间

W = torch.full((2,2), 2., requires_grad=True)
X = torch.full((2,2), 3., requires_grad=True)
B = torch.full((2,2), 4., requires_grad=True)

out = line.apply(W, X, B) # 调用line类的apply方法，传入W,X,B，返回y
out.backward(torch.ones(2,2))
print(W, X, B)
print(W.grad, X.grad, B.grad)