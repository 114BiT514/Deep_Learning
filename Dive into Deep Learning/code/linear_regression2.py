import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn

true_w = torch.tensor(eval(input("请输入权重：")),dtype=torch.float32)
true_b = float(eval(input("请输入偏置:")))
features,labels = d2l.synthetic_data(true_w,true_b,1000)    #生成数据集

def load_array(data_arrays,batch_size,is_train):    #数据迭代器
    dataset = data.TensorDataset(*data_arrays)  #将数据进行打包zip
    return data.DataLoader(dataset,batch_size,shuffle=is_train)

batch_size = 10
data_iter = load_array((features,labels),batch_size,True)

net = nn.Sequential(nn.Linear(2,1))
net[0].weight.data.normal_(0,0.01)
net[0].bias.data.fill_(0)

loss = nn.MSELoss()

trainer = torch.optim.SGD(net.parameters(),lr=0.03)

num_epochs = 3
for epoch in range(num_epochs):
    for X,y in data_iter:
        l = loss(net(X),y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    with torch.no_grad():
        L = loss(net(features),labels)
        print(f"epoch{epoch+1}:loss {L.item():.6f}")


