import torch
from models import ViViTBackbone
import numpy as np

def get_num_parameters(model):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in parameters]) / 1_000_000

ViViTModel3 = ViViTBackbone(
    t=32,
    h=64,
    w=64,
    patch_t=8,
    patch_h=4,
    patch_w=4,
    num_classes=10,
    dim=512,
    depth=6,
    heads=8,
    mlp_dim=8,
    model=3,
    device='cpu'
)
ViViTModel4 = ViViTBackbone(
    t=32,
    h=64,
    w=64,
    patch_t=8,
    patch_h=4,
    patch_w=4,
    num_classes=10,
    dim=512,
    depth=6,
    heads=8,
    mlp_dim=8,
    model=4,
    device='cpu'
)

device = torch.device('cpu')
vid = torch.rand(32, 3, 32, 64, 64).to(device)

pred_3 = ViViTModel3(vid) # (32, 10)
pred_4 = ViViTModel4(vid) # (32, 10)

param_3 = get_num_parameters(ViViTModel3)
param_4 = get_num_parameters(ViViTModel4)
print('Model 3 Trainable Parameters: %.3fM' % param_3)
print('Model 4 Trainable Parameters: %.3fM' % param_4)
