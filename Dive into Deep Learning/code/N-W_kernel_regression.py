import torch
from torch import nn
from d2l import torch as d2l

n_train = 50
x_train, _ = torch.sort((torch.rand(n_train) * 5))
def f(x):
    return 2 * torch.sin(x) + x ** 0.8

y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,))
x_test = torch.arange(0, 5, 0.1)
y_truth = f(x_test)
n_test = len(x_test)

def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5), cmap='Reds'):
    #输入matrices的形状是（要显示的行数，要显示的列数，查询的数目，键的数目）
    d2l.use_svg_display()
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    #返回一个 figure（画布），多个 axes（子图），子图共享同一个x轴y轴，而且不消除画布值为一的维度
    fig, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize, sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            #imshow 是 Matplotlib 中用于显示二维矩阵的函数,把一个矩阵画成热力图
            pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    #生成颜色条,shrink调整颜色条大小
    fig.colorbar(pcm, ax=axes, shrink=0.6)
    d2l.plt.show()

def plot_kernel_reg(y_hat):
    d2l.plot(x_test, [y_truth, y_hat], 'x', 'y',
             legend=['Truth', 'Pred'],xlim=[0, 5], ylim=[-1, 5])
    #'o'表示每个点用圆圈 'o' 表示.alpha=0.5表示设置透明度为 0.5
    d2l.plt.plot(x_train, y_train, 'o', alpha=0.5)
    d2l.plt.show()

# y_hat = torch.repeat_interleave(y_train.mean(),n_test)    #沿指定维度重复张量中的元素(按元素重复)
# plot_kernel_reg(y_hat)

# X_repeat = x_test.repeat_interleave(n_train).reshape(-1, n_train)
# attention_weights = nn.functional.softmax(-(X_repeat - x_train) ** 2 / 2, dim=1)
# y_hat = torch.matmul(attention_weights, y_train)
# plot_kernel_reg(y_hat)
# show_heatmaps(attention_weights.unsqueeze(0).unsqueeze(0),
#               xlabel='Sorted training inputs',ylabel='Sorted testing inputs')

# X = torch.ones((2, 1, 4))
# Y = torch.ones((2, 4, 6))
# print(torch.bmm(X, Y).shape)

# weights = torch.ones((2, 10)) * 0.1
# values = torch.arange(20.0).reshape((2, 10))
# print(torch.bmm(weights.unsqueeze(1), values.unsqueeze(-1)))

class NWKernelRegression(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = nn.Parameter(torch.rand((1, ), requires_grad=True))

    def forward(self, queries, keys, values):
        queries = queries.repeat_interleave(keys.shape[1]).reshape(-1, keys.shape[1])
        self.attention_weights = nn.functional.softmax(-((queries - keys) * self.w) ** 2 / 2, dim=1)
        return torch.bmm(self.attention_weights.unsqueeze(1), values.unsqueeze(-1)).reshape(-1)

X_tile = x_train.repeat((n_train, 1))
Y_tile = y_train.repeat((n_train, 1))

keys = X_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape(n_train, -1)
values = Y_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape(n_train, -1)

net = NWKernelRegression()
loss = nn.MSELoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
animator = d2l.Animator(xlabel='epoch', ylabel='loss',xlim=[1, 5])

for epoch in range(5):
    trainer.zero_grad()
    l = loss(net(x_train, keys, values), y_train)
    l.sum().backward()
    trainer.step()
    print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
    animator.add(epoch + 1, float(l.sum()))
d2l.plt.show()

keys = x_train.repeat((n_test, 1))
values = y_train.repeat((n_test, 1))
y_hat = net(x_test, keys, values).detach()
plot_kernel_reg(y_hat)

show_heatmaps(net.attention_weights.unsqueeze(0).unsqueeze(0),
              xlabel='Sorted training inputs', ylabel='Sorted testing inputs')




