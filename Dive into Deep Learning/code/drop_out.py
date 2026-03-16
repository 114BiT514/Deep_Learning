import torch
from torch import nn
import torchvision
from torchvision import transforms
from matplotlib_inline import backend_inline
from torch.utils import data
import matplotlib.pyplot as plt
import os

os.chdir('E:\Miniconda\envs\d2l')

def main():
    def dropout_layer(X, dropout):
        assert 0 <= dropout <= 1
        if dropout == 1:
            return torch.zeros_like(X)
        if dropout == 0:
            return X
        mask = (torch.rand(X.shape) > dropout).float()
        return mask * X /(1.0 - dropout)

    num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
    dropout1, dropout2 = 0.2, 0.5

    def get_dataloader_workers():
        return 4

    def load_data_fashion_mnist(batch_size, resize=None):
        """下载Fashion-MNIST数据集，然后将其加载到内存中"""
        trans = [transforms.ToTensor()]
        if resize:
            trans.insert(0, transforms.Resize(resize))
        trans = transforms.Compose(trans)
        mnist_train = torchvision.datasets.FashionMNIST(
            root="../data", train=True, transform=trans, download=True)
        mnist_test = torchvision.datasets.FashionMNIST(
            root="../data", train=False, transform=trans, download=True)
        return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                                num_workers=get_dataloader_workers()),
                data.DataLoader(mnist_test, batch_size, shuffle=False,
                                num_workers=get_dataloader_workers()))

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
            self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
            if nrows * ncols == 1:
                self.axes = [self.axes, ]
            # 使用lambda函数捕获参数(用lambda函数可以做到初始化时不会调用set_axes（），只有self.config_axes()才会调用)
            self.config_axes = lambda: set_axes(
                self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
            self.X, self.Y, self.fmts = None, None, fmts

        def add(self, x, y):        #self.X和self.Y是列表的列表，X[i]表示一条曲线；x,y一般是列表，表示新加入这些曲线的点
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
            plt.draw()
            plt.pause(0.5)

        def close(self):
            plt.show()

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

    class Net(nn.Module):
        def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2, is_training=True):
            super(Net, self).__init__()
            self.num_inputs = num_inputs
            self.training = is_training
            self.lin1 = nn.Linear(num_inputs, num_hiddens1)
            self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
            self.lin3 = nn.Linear(num_hiddens2, num_outputs)
            self.relu = nn.ReLU()

        def forward(self, X):
            H1 = self.relu(self.lin1(X.reshape((-1, num_inputs))))
            if self.training:
                H1 = dropout_layer(H1, dropout1)
            H2 = self.relu(self.lin2(H1))
            if self.training:
                H2 = dropout_layer(H2, dropout2)
            out = self.lin3(H2)
            return out

    # net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)
    num_epochs, lr, batch_size = 10, 0.5, 256
    loss = nn.CrossEntropyLoss()
    train_iter, test_iter = load_data_fashion_mnist(batch_size)

    net = nn.Sequential(nn.Flatten(),
                        nn.Linear(num_inputs, num_hiddens1),
                        nn.ReLU(),
                        nn.Dropout(dropout1),
                        nn.Linear(num_hiddens1, num_hiddens2),
                        nn.ReLU(),
                        nn.Dropout(dropout2),
                        nn.Linear(num_hiddens2, num_outputs),)

    def init_params(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)
            m.bias.data.fill_(0)

    net.apply(init_params)
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)


if __name__ == '__main__':
    main()







