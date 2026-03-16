import torch
from d2l import torch as d2l
import torchvision
from torch.utils import data
from torchvision import transforms
from matplotlib_inline import backend_inline
import os

os.chdir('E:\Miniconda\envs\d2l')

def main():
    batch_size = 256

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

    train_iter, test_iter = load_data_fashion_mnist(batch_size)

    num_inputs = 784
    num_outputs = 10

    W = torch.normal(0, 0.01, size=(num_inputs,num_outputs), requires_grad=True)
    b = torch.zeros(num_outputs, requires_grad=True)

    def softmax(X):
        X_exp = torch.exp(X)
        partition = X_exp.sum(axis=1, keepdim=True)     #规范化常数
        return X_exp / partition

    def net(X):
        return softmax(torch.matmul(X.reshape(-1,W.shape[0]), W) + b)

    def cross_entropy(y_hat, y):        #交叉熵损失
        return - torch.log(y_hat[range(len(y_hat)), y])

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

    def sgd(params, lr, batch_size):  # 定义优化算法：小批量随机梯度下降
        with torch.no_grad():
            for param in params:
                param -= lr * param.grad / batch_size
                param.grad.zero_()

    lr = 0.1
    num_epochs = 10

    def updater(batch_size):
        return sgd([W, b], lr, batch_size)

    train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)

    def get_fashion_mnist_labels(labels):  # 用于数字标签索引与文本名称之间的转换
        text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                       'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
        return [text_labels[int(i)] for i in labels]

    def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  # 用于样本可视化
        """绘制图像列表"""
        figsize = (num_cols * scale, num_rows * scale)
        _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
        """开头加 _ 纯粹是个惯例，不是 Python 语法特性。它的意思是：“我知道这里有个返回值，但我不打算用它”。"""
        axes = axes.flatten()
        for i, (ax, img) in enumerate(zip(axes, imgs)):
            if torch.is_tensor(img):
                # 图片张量
                ax.imshow(img.numpy())
            else:
                # PIL图片
                ax.imshow(img)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            if titles:
                ax.set_title(titles[i])
        return axes

    def predict_ch3(net, test_iter, n=6):   #预测标签
        for X, y in test_iter:
            break
        trues = get_fashion_mnist_labels(y)
        preds = get_fashion_mnist_labels(net(X).argmax(axis=1))
        titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
        show_images(X[0:n].reshape(n, 28, 28), 1, n, titles=titles[0:n])

    predict_ch3(net, test_iter)
    d2l.plt.show()

if __name__ == '__main__':
    main()