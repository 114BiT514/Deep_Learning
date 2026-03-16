import torch

def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():
    device = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return device if device else [torch.device('cpu')]

print(try_all_gpus(), try_gpu(), try_gpu(100))