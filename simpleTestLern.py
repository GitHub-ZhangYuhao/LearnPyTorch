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



'''
Python NNE Model Create (UE5.5 NNE )
https://dev.epicgames.com/community/learning/tutorials/7dr8/unreal-engine-nne-neural-post-processing#modelassetcreation
'''
import torch.nn as nn
class SobelFilter(torch.nn.Module):

    def __init__(self):
        super(SobelFilter, self).__init__()
        self.hFilter = torch.tensor([[[[1, 0, -1], [2, 0, -2], [1, 0, -1]]] * 3] * 3, dtype=torch.float)
        self.vFilter = torch.tensor([[[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]] * 3] * 3, dtype=torch.float)

    def forward(self, x):
        x = torch.nn.functional.pad(x, (1, 1, 1, 1), mode='replicate')
        h = torch.nn.functional.conv2d(x, self.hFilter)
        v = torch.nn.functional.conv2d(x, self.vFilter)
        return torch.sqrt(h * h + v * v)

if __name__=="__main__":
    model = SobelFilter()
    input = torch.randn(1, 3, 256, 256)
    onnx = torch.onnx.export(model, (input,), 'sobel.onnx',
                             input_names=['input'], output_names=['output'], opset_version=9,
                             dynamic_axes={'input': {2: 'height', 3: 'width'}, 'output' : {2: 'height', 3: 'width'}})
'''
End
'''