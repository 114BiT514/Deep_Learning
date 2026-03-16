import random
import torch
from d2l import torch as d2l
import numpy as np
from matplotlib_inline import backend_inline
import matplotlib.pyplot as plt

def use_svg_display():
    """使用svg格式在Jupyter中显示绘图"""
    backend_inline.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    """设置matplotlib的图表大小"""
    use_svg_display()
    d2l.plt.rcParams['figure.figsize'] = figsize

def synthetic_data(w,b,num_examples):   #用于生成数据集
    X = torch.normal(0,1,(num_examples,len(w)))   #从标准正态分布中抽样特征
    y = torch.matmul(X,w) + b   #生成标签
    y += torch.normal(0,0.01,y.shape)   #引入噪声项
    return X,y.reshape((-1,1))


true_w = torch.tensor(eval(input("请输入权重向量：")),dtype=torch.float32)
true_b = eval(input("请输入偏置："))
features,labels = synthetic_data(true_w,true_b,1000)
set_figsize()
d2l.plt.scatter(features[:,1].detach().numpy(),labels.detach().numpy(),1)
plt.show()
d2l.plt.scatter(features[:,0].detach().numpy(),labels.detach().numpy(),1)
plt.show()

def data_iter(batch_size,features,labels):  #读取数据集，迭代小批量
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0,num_examples,batch_size):
        batch_indices = torch.tensor(indices[i:min(i+batch_size,num_examples)])
        yield features[batch_indices],labels[batch_indices]

w = torch.normal(0,0.01,size=(2,1),requires_grad = True)    #初始化参数
b = torch.zeros(1,requires_grad = True)

def linreg(X,w,b):  #定义线性回归模型
    return torch.matmul(X,w) + b

def squared_loss(y_hat,y):  #定义损失函数
    return (y_hat - y.reshape(y_hat.shape))**2/2

def sgd(params,lr,batch_size):  #定义优化算法：小批量随机梯度下降
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad/batch_size
            param.grad.zero_()

lr = 0.03
num_epochs = 100
net = linreg
loss = squared_loss
batch_size = 10
for epoch in range(num_epochs):
    for X,y in data_iter(batch_size,features,labels):
        l = loss(net(X,w,b),y)
        l.sum().backward()
        sgd([w,b],lr,batch_size)
    with torch.no_grad():
        train_l = loss(net(features,w,b),labels)
        print(f"epoch{epoch+1},loss{float(train_l.mean())}")
print(true_w - w.reshape(true_w.shape))
print(true_b - b.item())
