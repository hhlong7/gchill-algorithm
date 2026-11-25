import torch
import torchvision
from torch import nn
from workloads.format import validate
from workloads.torchfx import import_model
import pytest


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



#q, k, v projections -> sdpa -> output projection, on a 1 x 4 x 8 input with 2 heads of 4
class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q, self.k, self.v, self.o = (nn.Linear(8, 8) for _ in range(4))

    def forward(self, x):
        heads = lambda t: t.view(1, 4, 2, 4).transpose(1, 2) #(batch, heads, seq, head dim)
        a = nn.functional.scaled_dot_product_attention(heads(self.q(x)), heads(self.k(x)), heads(self.v(x)))
        return self.o(a.transpose(1, 2).reshape(1, 4, 8))


def test_attention_exact():
    w = import_model(TinyAttention(), torch.randn(1, 4, 8), "attn")
    proj = {"m": 4, "n": 8, "k": 8}
    got = [(k.type, k.shape, k.deps) for k in w.kernels]
    assert got == [
        ("gemm", proj, ()), ("gemm", proj, ()), ("gemm", proj, ()), #q, k, v
        ("bgemm", {"b": 2, "m": 4, "n": 4, "k": 4}, (0, 1)), #q @ k^T waits for q and k only
        ("elementwise", {"n": 2 * 4 * 4}, (3,)), #softmax
        ("bgemm", {"b": 2, "m": 4, "n": 4, "k": 4}, (4, 2)), #p @ v
        ("gemm", proj, (5,)),
    ]


#bert-base, sequence 128: 12 layers x (6 gemms, 2 bgemms, 6 elementwise) + embeddings and pooler
def test_bert_base():
    pytest.importorskip("transformers")
    from workloads.torchfx import bert_base
    w = import_model(*bert_base(128), "bert")
    validate(w)
    types = [k.type for k in w.kernels]
    assert (len(types), types.count("gemm"), types.count("bgemm"), types.count("elementwise")) == (176, 73, 24, 79)
    assert w.kernels[9].shape == {"b": 12, "m": 128, "n": 128, "k": 64} #layer 0 q @ k^T: 12 heads of 64
    assert w.kernels[9].deps == (6, 7)
    assert [k.id for k in w.kernels if not k.deps] == [0, 1, 3] #word, token type and position embeddings