import torch
from model import QuaternionTransformerNetworkTiny


DEVICE = 'cuda'
K = 16
C = 12

network = QuaternionTransformerNetworkTiny(patch_size=K, channels=C).to(DEVICE)

x = torch.randn(1, C, 256, 256).to(DEVICE)
output = network(x)
print(output.shape)
