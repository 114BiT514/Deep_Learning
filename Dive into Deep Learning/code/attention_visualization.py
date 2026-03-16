import torch
from d2l import torch as d2l

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

#toech.eye生成一个单位矩阵
attention_weight = torch.eye(10).reshape(1, 1, 10, 10)
show_heatmaps(attention_weight, xlabel='Keys', ylabel='Queries')



