import torch
from models import ViViTBackbone

v = ViViTBackbone(
    t=32,
    h=64,
    w=64,
    patch_t=8,
    patch_h=4,
    patch_w=4,
    num_classes=10,
    dim=512,
    depth=6,
    heads=10,
    mlp_dim=8,
    model=3
)

device = torch.device('cpu')
vid = torch.rand(32, 3, 32, 64, 64).to(device)

pred = v(vid)  # (32, 10)

parameters = filter(lambda p: p.requires_grad, v.parameters())
parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
print('Trainable Parameters: %.3fM' % parameters)
