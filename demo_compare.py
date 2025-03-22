import torch

'''
比较两个tensor是否相等
要进行比较需要两个tensor的shape相同
'''
a = torch.rand(2, 3)
b = torch.rand(2, 3)

print(a)
print(b)

print(torch.eq(a, b)) #逐元素比较
print(torch.equal(a, b)) #比较两个tensor是否相等

print(torch.ge(a, b)) #逐元素比较 >=
print(torch.gt(a, b)) #逐元素比较 >
print(torch.le(a, b)) #逐元素比较 <=
print(torch.lt(a, b)) #逐元素比较 <
print(torch.ne(a, b)) #逐元素比较 !=

'''
tensor的排序
'''
a = torch.tensor([1, 4, 4, 3, 5])
print(torch.sort(a)) #返回了排序后的结果和对应原始数列中的索引 同a.sort()
# values=tensor([1, 3, 4, 4, 5]),
# indices=tensor([0, 3, 1, 2, 4]))
print(torch.sort(a, descending=True)) #指定为 降序排序

a = torch.tensor( [[1, 4, 4, 3, 5] ,
                   [2, 3, 1, 3, 5]])
print(a.shape)
'''torch.Size([2, 5])'''
print(torch.sort(a, dim=0, descending=True)) #指定维度为0 降序排序 ，对每一列进行排序
'''values=tensor([[2, 4, 4, 3, 5],
                  [1, 3, 1, 3, 5]]),'''
print(torch.sort(a, dim=1, descending=True)) #指定维度为1 降序排序 ， 对每一行进行排序（对每个数列进行排序）
'''values=tensor([[5, 4, 4, 3, 1],
                  [5, 3, 3, 2, 1]]),'''

'''
tensor的topk
'''
#topk 返回前k个最大的元素,会按照降序排序
a = torch.tensor([[2, 4, 3, 1, 5],
                  [1, 3, 4, 5, 2]]) # 2*5
print(a.shape)
print(torch.topk(a, k=2 , dim=0)) #返回每一列的前两个最大值
print(torch.topk(a, k=2 , dim=1)) #返回每一行的前两个最大值
#dim=0 可以理解为从 Shape 的 0 的结果也就是 2 为列向量 进行排序
#dim=1 可以理解为从 Shape 的 1 的结果也就是 5 为行向量 进行排序

'''
tensor的kthvalue
'''
#kthvalue 返回第k小的元素
print(torch.kthvalue(a, k=2, dim=0)) #返回每一列的第2小的数
print(torch.kthvalue(a, k=1, dim=0)) #返回每一行的第1小的数
print(torch.kthvalue(a, k=2, dim=1)) #返回每一行的第2小的数
print(torch.kthvalue(a, k=1, dim=1)) #返回每一行的第1小的数

'''
Tensor判断是否有界
'''
a = torch.rand(2,3)
print(a)
print(a/0)
print(torch.isfinite(a)) #判断是否有界
print(torch.isfinite(a/0)) #判断是否有界
print(torch.isinf(a/0)) #判断是否无穷大
print(torch.isnan(a/0)) #判断是否为nan

import numpy as np
a = torch.tensor([1,2,np.nan]) #使用numpy定义nan
print(torch.isnan(a)) #判断是否为nan
