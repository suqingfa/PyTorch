import torch

print('PyTorch version:', torch.__version__)
print('cuda is available:', torch.cuda.is_available())
print('cuda device count:', torch.cuda.device_count())
