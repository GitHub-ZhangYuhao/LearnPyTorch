import torch

''' 定义 tensor 的数据类型 和 设备  '''
dev = torch.device('cpu') #定义设备为cpu,将张量放在cpu上计算
dev = torch.device('cuda') #定义设备为cuda,将张量放在第0块显卡上cuda上计算
#print(torch.cuda.is_available())
a = torch.tensor([2,2], dtype=torch.float32, device=dev) #创建一个张量，值为2,2，存放设备为cpu/cuda
#cpu : tensor([2, 2])
#cude : tensor([2, 2], device='cuda:0')
print(a)

''' 定义稀疏的张量 '''
i = torch.tensor([ [0,1,2],[0,1,2] ]) #定义稀疏张量的行索引,将对角线定义为非0元素，也就是索引 (0,0),(1,1),(2,2) 定义为非0元素
v = torch.tensor([1,2,3]) #定义稀疏张量的非0元素
'''
即:tensor([[1, 0, 0, 0],
           [0, 2, 0, 0],
           [0, 0, 3, 0]
           [0, 0, 0, 0]])
'''
#a = torch.sparse_coo_tensor(i, v, (4,4)) #创建一个稀疏张量，大小为4*4，非0元素为1,2,3，索引为(0,0),(1,1),(2,2)
a = torch.sparse_coo_tensor(i, v, (4,4),
                            dtype=torch.float32,
                            device=dev).to_dense() #创建一个稀疏张量，大小为4*4，非0元素为1,2,3，索引为(0,0),(1,1),(2,2)
print(a)