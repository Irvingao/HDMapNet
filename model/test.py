from base import BevTransformerEncoder
import torch
model = BevTransformerEncoder(64, 4)
model.cuda()

x = torch.randn(1, 64, 200, 400).cuda()
print(model(x).shape) # (1, 256, 64, 64)