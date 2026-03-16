import torch
from torch import nn
import numpy as np
from matplotlib_inline import backend_inline
from d2l import torch as d2l
from torch.utils import data

def use_svg_display():
    """使用svg格式在Jupyter中显示绘图"""
    backend_inline.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    """设置matplotlib的图表大小"""
    use_svg_display()
    d2l.plt.rcParams['figure.figsize'] = figsize

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置matplotlib的轴"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',fmts=('-', 'm--', 'g-.', 'r:'),
         figsize=(3.5, 2.5), axes=None):
    """绘制数据点"""
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else d2l.plt.gca()

    # 如果X有一个轴，输出True
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)

def load_array(data_arrays,batch_size,is_train):    #数据迭代器
    dataset = data.TensorDataset(*data_arrays)  #将数据进行打包zip
    return data.DataLoader(dataset,batch_size,shuffle=is_train)

T = 1000
time = torch.arange(1, T + 1, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))
d2l.plt.show()

tau = 4
features = torch.zeros((T - tau, tau))
for i in range(tau):
    features[:,i] = x[i:T - tau + i]
labels = x[tau:].reshape(-1,1)
batch_size, n_train = 16, 600
train_iter = load_array((features[:n_train], labels[:n_train]), batch_size, is_train=True)

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

def get_net():
    net = nn.Sequential(nn.Linear(4, 10), nn.ReLU(), nn.Linear(10,1))
    net.apply(init_weights)
    return net

loss = nn.MSELoss(reduction='none')

def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.sum().backward()
            trainer.step()
        print(f'epoch {epoch + 1}, '
              f'loss: {d2l.evaluate_loss(net, train_iter, loss):f}')

net = get_net()
train(net, train_iter, loss, 5, 0.01)

onestep_preds = net(features)
# plot([time, time[tau:]],
#          [x.detach().numpy(), onestep_preds.detach().numpy()], 'time',
#          'x', legend=['data', '1-step preds'], xlim=[1, 1000],
#          figsize=(6, 3))
#
# d2l.plt.show()

# multistep_preds = torch.zeros(T)
# multistep_preds[:n_train + tau] = x[:n_train + tau]
# for i in range(n_train + tau, T):
#     multistep_preds[i] = net(multistep_preds[i - tau:i].reshape(1, -1))
#
# plot([time, time[tau:], time[n_train + tau:]],
#          [x.detach().numpy(), onestep_preds.detach().numpy(),
#           multistep_preds[n_train + tau:].detach().numpy()], 'time',
#          'x', legend=['data', '1-step preds', 'multistep preds'],
#          xlim=[1, 1000], figsize=(6, 3))
#
# d2l.plt.show()

max_steps = 64
features = torch.zeros((T - tau - max_steps + 1,tau + max_steps))
for i in range(tau):
    features[:,i] = x[i:i + T - tau - max_steps + 1]

for i in range(tau, tau + max_steps):
    features[:,i] = net(features[:,i - tau : i]).reshape(-1)

steps = (1, 4, 16, 64)
plot([time[tau + i - 1: T - max_steps + i] for i in steps],
         [features[:, (tau + i - 1)].detach().numpy() for i in steps], 'time', 'x',
         legend=[f'{i}-step preds' for i in steps], xlim=[5, 1000],
         figsize=(6, 3))

d2l.plt.show()




