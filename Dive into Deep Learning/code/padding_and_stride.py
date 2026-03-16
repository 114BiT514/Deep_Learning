import torch
from torch import nn

# 为了方便起见，我们定义了一个计算卷积层的函数。
# 此函数初始化卷积层权重，并对输入和输出提高和缩减相应的维数
def comp_conv2d(conv2d, X):
    X = X.reshape((1,1) + X.shape)
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:])

conv2d = nn.Conv2d(1, 1, kernel_size=(3, 3), padding=1)
X = torch.rand(size=(8,8))
Y = comp_conv2d(conv2d, X)
print(Y,Y.shape)

conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))
X = torch.rand(size=(8,8))
Y = comp_conv2d(conv2d, X)
print(Y,Y.shape)

conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
X = torch.rand(size=(8,8))
Y = comp_conv2d(conv2d, X)
print(Y,Y.shape)

conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3,4))
X = torch.rand(size=(8,8))
Y = comp_conv2d(conv2d, X)
print(Y,Y.shape)

