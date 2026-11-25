import torch
import torchvision
from torch import nn
from workloads.format import validate
from workloads.torchfx import import_model


#conv -> relu -> flatten -> linear on a 2 x 3 x 4 x 4 input: flatten makes no kernel, so linear waits for relu
class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(8 * 4 * 4, 10)

    def forward(self, x):
        return self.fc(torch.flatten(self.relu(self.conv(x)), 1))


def test_tiny_exact():
    w = import_model(Tiny(), torch.randn(2, 3, 4, 4), "tiny")
    got = [(k.type, k.shape, k.deps) for k in w.kernels]
    assert got == [
        ("gemm", {"m": 2 * 4 * 4, "n": 8, "k": 3 * 3 * 3}, ()), #im2col: batch x out h x out w, in ch x 3 x 3
        ("elementwise", {"n": 2 * 8 * 4 * 4}, (0,)),
        ("gemm", {"m": 2, "n": 10, "k": 128}, (1,)),
    ]


#resnet-50: 53 convs + 1 linear = 54 gemms, 53 bn + 49 relu + 16 adds = 118 elementwise, 2 pools
def test_resnet50():
    w = import_model(torchvision.models.resnet50(weights=None), torch.randn(1, 3, 224, 224), "r50")
    validate(w)
    types = [k.type for k in w.kernels]
    assert (len(types), types.count("gemm"), types.count("elementwise"), types.count("reduce")) == (174, 54, 118, 2)
    assert w.kernels[0].shape == {"m": 112 * 112, "n": 64, "k": 3 * 7 * 7} #conv1: 7x7 stride 2
    assert sum(1 for k in w.kernels if len(k.deps) == 2) == 16 #every residual add joins 2 branches
    assert [k.id for k in w.kernels if not k.deps] == [0] #one entry point