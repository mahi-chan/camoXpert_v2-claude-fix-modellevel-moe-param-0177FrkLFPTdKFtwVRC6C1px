import torch
import torch.nn as nn
from models.frequency_expert import DeepWaveletDecomposition, HighFrequencyAttention, LowFrequencyAttention, ODEEdgeReconstruction

feature_dims = [64, 128, 320, 512]
total = 0

print('FEDER Component Parameter Breakdown:')
print('='*60)

# Wavelets
wavelet_params = 0
for dim in feature_dims:
    w = DeepWaveletDecomposition(dim)
    p = sum(p.numel() for p in w.parameters())
    wavelet_params += p
print(f'Wavelets (4 scales): {wavelet_params/1e6:.2f}M')
total += wavelet_params

# High-freq attention
high_params = 0
for dim in feature_dims:
    h = HighFrequencyAttention(dim)
    p = sum(p.numel() for p in h.parameters())
    high_params += p
    print(f'  HighFreqAtt dim={dim}: {p/1e6:.2f}M')
print(f'HighFreqAtt (4 scales): {high_params/1e6:.2f}M')
total += high_params

# Low-freq attention
low_params = 0
for dim in feature_dims:
    l = LowFrequencyAttention(dim)
    p = sum(p.numel() for p in l.parameters())
    low_params += p
print(f'LowFreqAtt (4 scales): {low_params/1e6:.2f}M')
total += low_params

# ODE
ode_params = 0
for dim in feature_dims:
    o = ODEEdgeReconstruction(dim)
    p = sum(p.numel() for p in o.parameters())
    ode_params += p
print(f'ODE (4 scales): {ode_params/1e6:.2f}M')
total += ode_params

# Fusion
fusion_params = 0
for dim in feature_dims:
    fusion = nn.Sequential(
        nn.Conv2d(dim * 4, dim, 3, padding=1, bias=False),
        nn.BatchNorm2d(dim),
        nn.ReLU(inplace=True),
        nn.Conv2d(dim, dim, 3, padding=1, bias=False),
        nn.BatchNorm2d(dim),
        nn.ReLU(inplace=True)
    )
    p = sum(p.numel() for p in fusion.parameters())
    fusion_params += p
print(f'Fusion (4 scales): {fusion_params/1e6:.2f}M')
total += fusion_params

print('='*60)
print(f'TOTAL (without decoder): {total/1e6:.2f}M')

# Full expert
from models.expert_architectures import FEDERFrequencyExpert
feder = FEDERFrequencyExpert(feature_dims)
feder_total = sum(p.numel() for p in feder.parameters())
print(f'FULL FEDER Expert: {feder_total/1e6:.2f}M')
print(f'Decoder: {(feder_total - total)/1e6:.2f}M')
