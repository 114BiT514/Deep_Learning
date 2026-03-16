import torch
from torch import nn
from d2l import torch as d2l
import os
from matplotlib_inline import backend_inline

os.chdir('E:\Miniconda\envs\d2l')

def main():
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

    def init_net(m):
        if type(m) == nn.Linear:
            m.weight.data.normal_(0, 0.01)
            m.bias.data.fill_(0)

    net.apply(init_net)

    loss = nn.CrossEntropyLoss(reduction='none')

    trainer = torch.optim.SGD(net.parameters(), lr=0.1)

    def accuracy(y_hat, y):         #预测正确的数量    #函数自动创建了一个新引用，指向我传的这个变量引用的地址
        if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:     #y_hat得是个矩阵
            y_hat = y_hat.argmax(axis=1)
        cmp = y_hat.type(y.dtype) == y
        return float(cmp.type(y.dtype).sum())

    class Accumulator:      #用于在n个变量上累加
        def __init__(self, n):
            self.data = [0.0] * n

        def add(self, *args):
            self.data = [a + float(b) for a, b in zip(self.data, args)]

        def reset(self):
            self.data = [0.0] * len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    def evaluate_accuracy(net,data_iter):       #计算在指定数据集上net模型的精度
        if isinstance(net, torch.nn.Module):        #判断一个对象是不是某个类的实例
            net.eval()      #将模型设置为评估模式
        metric = Accumulator(2)      #存正确预测数和预测总数
        with torch.no_grad():
            for X, y in data_iter:
                metric.add(accuracy(net(X), y), y.numel())
        return metric[0] / metric[1]

    def train_epoch_ch3(net, train_iter, loss, updater):    #训练模型一轮
        if isinstance(net, torch.nn.Module):        #将模型设置为训练模式
            net.train()
        metric = Accumulator(3)     #训练损失总和，训练精确度总和，样本数
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y)
            if isinstance(updater, torch.optim.Optimizer):
                updater.zero_grad()
                l.mean().backward()
                updater.step()
            else:
                l.sum().backward()
                updater(X.shape[0])
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())   #累加的是本次batch训练之前的
        return metric[0] / metric[2], metric[1] / metric[2]         #返回的是本轮的平均损失和平均精度

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


    def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
        animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                            legend=['train loss', 'train acc', 'test acc'])
        for epoch in range(num_epochs):
            train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
            test_acc = evaluate_accuracy(net, test_iter)
            animator.add(epoch + 1, train_metrics + (test_acc,))
        train_loss, train_acc = train_metrics
        assert train_loss < 0.5, train_loss
        assert train_acc <= 1 and train_acc > 0.7, train_acc
        assert test_acc <= 1 and test_acc > 0.7, test_acc
        animator.close()

    num_epochs = 10
    train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

if __name__ == '__main__':
    main()

