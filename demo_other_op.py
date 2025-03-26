import torch
import numpy as np
import cv2

'''
这里使用CV2读取图片，
然后将图片的结果从numpy转换为tensor，
将tensor放到GPU上，
然后使用torch的flip函数进行翻转，
将处理后的tensor转换为numpy，
最后使用CV2显示图片
'''

data = cv2.imread("test.png")#读取图片
print(data)
#cv2.imshow("test",data) #显示图片
#cv2.waitKey(0) #等待按键

#a = np.zeros([2,2]) #生成一个2*2的矩阵，每个元素的值为0
out = torch.from_numpy(data) #将a转换为tensor
print(out)

out = out.to(torch.device("cuda")) #将out数据放到GPU上

print("CUDA 是否可用: ", torch.cuda.is_available(),
      "\n Out 是否再GPU上: ", out.is_cuda) #判断out是否在GPU上

out = torch.flip(out, dims=[0]) #将第0维的顺序翻转

#对于GPU上的数据无法直接转化到numpy, 需要先将数据放到CPU上
out = out.to(torch.device("cpu")) #将out数据放到CPU上
print("\n Out 是否在CPU上: ",out.is_cpu)

data = out.numpy() #将out从tensor转换为numpy数组
cv2.imshow("test",data) #显示图片
cv2.waitKey(0) #等待按键

'''可以用OpenCV导入图片，然后转换为tensor，进行处理'''