from thop import profile, clever_format
from net.gbnet import Net
import torch
import sys

model = Net()
inputs = torch.randn(4, 3,704, 704)

# Measure FLOPs and parameters
flops, params = profile(model, inputs=(inputs,))
flops, params = clever_format([flops, params], "%.3f")


print('[Statistics Information]\nFLOPs: {}\nParams: {}'.format(flops, params))

