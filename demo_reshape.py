import torch

a = torch.rand(2,3)
print(a)
out = torch.reshape(a, (3,2))
print(out)
'''
打印结果:
tensor([[0.8746, 0.4514, 0.9781],
        [0.7088, 0.4888, 0.9811]])
tensor([[0.8746, 0.4514],
        [0.9781, 0.7088],
        [0.4888, 0.9811]])
'''

#转置, 绕着对角线翻转
print(torch.t(out))

#交换两个维度,对于二维矩阵，交换第0和第1维度，结果也等同转置
print(torch.transpose(out, 0, 1)) #交换第0和第1维度

a = torch.linspace(1,6,6).view(1,2,3) #torch.rand(1,2,3)
print("a : ", a)
print(torch.transpose(a, 0,1))
print(torch.transpose(a, 0,1).shape)
#打印结果:(2,1,3)

out = torch.squeeze(a)#去掉维度为1的维度 ,去掉shape中的1
print(out)
print(out.shape)
#打印结果:(2,3)

out = torch.unsqueeze(a, -1) #在最后一个维度上添加一个维度, -1表示最后
print(out)
print(out.shape)
# 打印结果: torch.Size([1, 2, 3, 1])

out = torch.unbind(a, dim=1) # 切分第一个维度, 切分后返回一个元组,如：原本是(1,2,3)，切分后返回两个(1,3)
print(out)
# 打印结果: (tensor([[1., 2., 3.]]), tensor([[4., 5., 6.]]))

print(a)
#可以传递多个维度的翻转
print(torch.flip(a, [1,2])) # 翻转第1维度, 翻转后返回一个新的tensor ,如：原本是（1，2，3），dim=[1]，表示对行进行翻转
# 如果是 dim=[1,2],表示对行和列的内容都进行翻转
'''
dim = [1]
tensor([[4., 5., 6.],
        [1., 2., 3.])
dim = [1,2]
tensor([[[6., 5., 4.],
         [3., 2., 1.]]])
'''

print("torch.rot90")
out = torch.rot90(a,k=1)
print(out) # 旋转90度
print(out.shape) #torch.Size([2, 1, 3]) ,从(1,2,3)变成(2,1,3)