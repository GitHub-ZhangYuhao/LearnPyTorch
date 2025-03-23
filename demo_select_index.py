import torch

'''
#torch.where
'''
a = torch.rand(4,4)
b = torch.rand(4,4)
print(a)
print(b)

out = torch.where( a>0.5, a, b) # 如果a的值大于0.5的取a，否则取b（逐元素）
print(out)

'''
#torch.index_select
'''
print("torch.index_select")
a = torch.rand(4,4)
print(a)

# 按照行索引选择元素（筛选每列的，按照0，3，2顺序排列）,留下第0行，第3行，第2行
out = torch.index_select(a, dim=0,
             index=torch.tensor([0, 3, 2]) )
print(out)

'''
#torch.gather
'''
print("torch.gather")
a = torch.linspace(0, 15, 16).view(4, 4)#生成一个4*4的矩阵，元素的值为0到16
print(a)

out = torch.gather(a, dim=0,
             index=torch.tensor([[0,1,1,1],
                                 [0,1,2,2],
                                 [0,1,3,3]]))
print(out)
#如果 dim=0,
# 公式表示:out[i,j] = input[index[i,j], j]
# 如同:(out[i,j,k] = input[index[i,j,k], j, k])
'''
#dim=0, out[i, j, k] = input[index[i, j, k],         j,             k]
#dim=1, out[i, j, k] = input[i,         index[i, j, k],             k]
#dim=2, out[i, j, k] = input[i,                      j, index[i, j, k]]
'''
# 打印结果：
# tensor([[ 0.,  1.,  2.,  3.],
#         [ 4.,  5.,  6.,  7.],
#         [ 8.,  9., 10., 11.],
#         [12., 13., 14., 15.]])
# tensor([[ 0.,  5.,  6.,  7.],
#         [ 0.,  5., 10., 11.],
#         [ 0.,  5., 14., 15.]])


'''
#torch.masked_index
'''
print("torch.masked_index")
a = torch.linspace(0, 15, 16).view(4, 4)#生成一个4*4的矩阵，元素的值为0到16
mask = torch.gt(a, 8)
print(mask)
out = torch.masked_select(a, mask) # 按照mask的条件选择元素,将选中的值放在一个一维张量中
print(out)

'''
#torch.take
'''
print("torch.take")
a = torch.linspace(0, 15, 16).view(4, 4)#生成一个4*4的矩阵，元素的值为0到16
out = torch.take(a, torch.tensor([0, 15, 14, 10]))  #按照索引选择元素,将选中的值放在一个一维张量中
print(out)
# 打印结果: tensor([ 0., 15., 14., 10.])

'''
#torch.nonzero
稀疏表示很有用
'''
print("torch.nonzero")
a = torch.tensor([[0, 1, 2, 0],
                  [3, 0, 0, 5],
                  [0, 1, 0, 2]])
out = torch.nonzero(a) # 输出非0元素的 索引 (行,列)
print(out)
#打印结果:tensor([[0, 1],
            #   [0, 2],
            #   [1, 0],
            #   [1, 3],
            #   [2, 1],
            #   [2, 3]])