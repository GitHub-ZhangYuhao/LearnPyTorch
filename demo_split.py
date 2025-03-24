import torch
'''
torch.chunk() 函数可以将一个张量分割成多个张量
'''
print("torch.chunk")
#a = torch.rand((3,4))
a = torch.linspace(1,12,12).view(3,4)
# (tensor([[1., 2., 3.,  4.],
#          [5., 6., 7.,  8.],
#          [9., 10.,11., 12.]])

out = torch.chunk(a, 2, dim=0) #dim = 0,表示为每行向量为单位进行分割
print(out)
out = torch.chunk(a, 2, dim=1) #dim = 1,表示为每列向量为单位进行分割
print(out)

# (tensor([[1., 2., 3., 4.],
#         [5., 6., 7., 8.]]),
#  tensor([[ 9., 10., 11., 12.]]))
#
# (tensor([[ 1.,  2.],
#         [ 5.,  6.],
#         [ 9., 10.]]),
#  tensor([[ 3.,  4.],
#          [ 7.,  8.],
#          [11., 12.]]))

'''
torch.split() 函数可以将一个张量分割成多个张量
这种切分方式比较常用，可以包含chunk的功能
'''
print("torch.split")
a = torch.linspace(1,40,40).view(10,4)
print(a)
# 第一种切分方式,固定分割成3行
out = torch.split(a, 3, dim=0)
print(out)
print(len(out))

# 第二种切分方式，按照一个list进行分割
out = torch.split(a, [1,3,6], dim=0) #如果dim = 0，列表的和需要等于 行 的size，如果dim=1，列表的和要等 列 size
'''
输出的 尺寸分别为：
 1x4
 3x4
 6x4
'''
print(out)
print(len(out))
for t in out:
    print(t)