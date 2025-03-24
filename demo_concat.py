import torch

a = torch.zeros((2,4))
b = torch.ones((2,4))

'''
torch.cat() 函数可以将两个张量拼接在一起
'''
print("torch.cat")
#dim表示要修改输出的维度，dim=0表示修改行，dim=1表示修改列
print(torch.cat((a,b), dim=0)) #dim=0 表示在第0维拼接，即修改输出的行拼接
#输出shape为(4,4)
print(torch.cat((a,b), dim=1)) #dim=1 表示在第1维拼接，即修改输出的列拼接
#输出shape为(2,8)

'''
torch.stack() 函数可以将两个张量拼接在一起
'''
print("torch.stack")
a = torch.linspace(1,6,6).view(2,3) #生成一个2*3的矩阵，元素为1到6
b = torch.linspace(7,12,6).view(2,3) #生成一个2*3的矩阵，元素为7到12
print(a,b)
#dim 可以理解为从那个维度上开始拼接，0的话就是把a和b分别当作一个整体拼接，1的话就是把a和b的每一行当作一个整体拼接，2的话就是把a和b的每一行的每一个元素当作一个整体拼接
out = torch.stack((a,b), dim=0) #dim 表示输出要修改的维度，out[:,:,:],表示输出的第0维，第1维，第2维
print(out)
print(out.shape)    #得到 2*2*3 的shape
out = torch.stack((a,b), dim=1)
print(out)
print(out.shape)    #得到 2*2*3 的shape
print(out[:,0,:])   #得到第一个矩阵
print(out[:,1,:])   #得到第二个矩阵
out = torch.stack((a,b), dim=2)
print(out)
print(out.shape)    #得到 2*3*2 的shape
print(out[:,:,0])   #得到第一个矩阵
print(out[:,:,1])   #得到第二个矩阵
'''
out[:,1,:] 是一种切片操作。切片操作允许你从数组或张量中选取特定的元素。
: 表示选取该维度上的所有元素。
1 表示选取该维度上索引为1的元素（索引从0开始）。

out[:,1,:] 的具体含义如下：
第一个 : 表示选取第一个维度（即最外层维度）上的所有元素。
1 表示选取第二个维度上索引为1的元素。
最后一个 : 表示选取第三个维度上的所有元素。
因此，out[:,1,:] 会返回一个二维张量，形状为 (2, 3)，其中包含了 out 中第二个维度上索引为1的所有元素。
'''