import torch

import pyrad.cuda_v2 as pyrad_cuda

device = "cuda:3"
data = torch.randn((2**3), device=device)
threshold = 0.5
bitfield = pyrad_cuda.packbits(data, threshold)
print(data)
print(bitfield)
