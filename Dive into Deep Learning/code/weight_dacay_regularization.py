import torch
from torch import nn
from d2l import torch as d2l
from torch.utils import data
from matplotlib_inline import backend_inline

def synthetic_data(w,b,num_examples):   #用于生成数据集
    X = torch.normal(0,1,(num_examples,len(w)))   #从标准正态分布中抽样特征
    y = torch.matmul(X,w) + b   #生成标签
    y += torch.normal(0,0.01,y.shape)   #引入噪声项
    return X,y.reshape((-1,1))

def load_array(data_arrays,batch_size,is_train):    #数据迭代器
    dataset = data.TensorDataset(*data_arrays)  #将数据进行打包zip
    return data.DataLoader(dataset,batch_size,shuffle=is_train)

n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
train_data = synthetic_data(true_w, true_b, n_train)
train_iter = load_array(train_data, batch_size, is_train=True)
test_data = synthetic_data(true_w, true_b, n_test)
test_iter = load_array(test_data, batch_size, is_train=False)

def init_params():
    w = torch.normal(0,1,size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1,requires_grad=True)
    return [w, b]

def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2

def linreg(X,w,b):  #定义线性回归模型
    return torch.matmul(X,w) + b

def squared_loss(y_hat,y):  #定义损失函数
    return (y_hat - y.reshape(y_hat.shape))**2/2

def use_svg_display():
    """使用svg格式在Jupyter中显示绘图"""
    backend_inline.set_matplotlib_formats('svg')

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

class Animator:
    """在动画中绘制数据"""

    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数(用lambda函数可以做到初始化时不会调用set_axes（），只有self.config_axes()才会调用)
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):  # self.X和self.Y是列表的列表，X[i]表示一条曲线；x,y一般是列表，表示新加入这些曲线的点
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        d2l.plt.draw()
        d2l.plt.pause(0.1)

    def close(self):
        d2l.plt.show()

def sgd(params,lr,batch_size):  #定义优化算法：小批量随机梯度下降
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad/batch_size
            param.grad.zero_()

class Accumulator:  # 用于在n个变量上累加
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def evaluate_loss(net, data_iter, loss):    #用于评估模型在给定数据集上的损失
    metric = d2l.Accumulator(2)
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), y.numel())
    return metric[0]/metric[1]

def train(lambd):
    w, b = init_params()
    net, loss = lambda X: linreg(X, w, b), squared_loss
    num_epochs, lr = 300, 0.003
    animator = Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y) + lambd * l2_penalty(w)    #此处用到了广播机制
            l.sum().backward()
            sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),
                                     evaluate_loss(net, test_iter, loss)))
    animator.close()
    print('w的L2范数是：', torch.norm(w).item())

#weight_decay 的简洁实现
def train_concise(wd):
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        param.data.normal_()
    loss = nn.MSELoss()
    num_epochs, lr = 1000, 0.003
    trainer = torch.optim.SGD([{"params" : net[0].weight, 'weight_decay' : wd}, {"params" : net[0].bias}], lr = lr)
    animator = Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.mean().backward()
            trainer.step()
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1,(evaluate_loss(net, train_iter, loss),evaluate_loss(net, test_iter, loss)))
    animator.close()
    print('w的L2范数：', net[0].weight.norm().item())

train_concise(10)







