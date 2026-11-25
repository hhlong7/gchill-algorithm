"""
torch.fx importer: an open model's real operator graph -> a workload
- symbolic_trace gives every operator and what it reads; ShapeProp runs one example input through it so
  every node knows its output shape
- conv2d -> gemm by im2col: m = batch x out h x out w, k = in channels x kernel h x kernel w, n = out channels
- linear -> gemm: m = all leading dims, k = in features, n = out features
- batchnorm, relu, add -> elementwise over the output; pooling -> reduce over the input
- flatten / view / reshape move no data: skipped, their users wait for what they waited for
anything else raises, so a new model never gets silently wrong kernels
run from the repo root: python -m workloads.torchfx
"""

import math
import operator
import os
import torch
import torchvision
from torch import nn
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
from workloads.format import Kernel, Workload, save

ELEMENTWISE_MODULES = (nn.BatchNorm2d, nn.ReLU)
POOL_MODULES = (nn.MaxPool2d, nn.AdaptiveAvgPool2d)
ELEMENTWISE_FUNCTIONS = (operator.add, torch.add, torch.relu) #resnet's "out += identity" traces as operator.add
NO_DATA_FUNCTIONS = (torch.flatten,)
NO_DATA_METHODS = ("view", "reshape", "flatten", "contiguous")


def out_shape(node):
    return tuple(node.meta["tensor_meta"].shape)


def numel(node):
    return math.prod(out_shape(node))


#one fx node -> (kernel type, shape), or None if it moves no data
def kernel_of(node, gm):
    if node.op == "call_module":
        mod = gm.get_submodule(node.target)
        if isinstance(mod, nn.Conv2d):
            if mod.groups != 1:
                raise NotImplementedError(f"{node.name}: grouped conv")
            b, _, oh, ow = out_shape(node)
            kh, kw = mod.kernel_size
            return "gemm", {"m": b * oh * ow, "n": mod.out_channels, "k": mod.in_channels * kh * kw}
        if isinstance(mod, nn.Linear):
            return "gemm", {"m": numel(node) // mod.out_features, "n": mod.out_features, "k": mod.in_features}
        if isinstance(mod, ELEMENTWISE_MODULES):
            return "elementwise", {"n": numel(node)}
        if isinstance(mod, POOL_MODULES):
            return "reduce", {"n": numel(node.all_input_nodes[0])}
    if node.op == "call_function":
        if node.target in ELEMENTWISE_FUNCTIONS:
            return "elementwise", {"n": numel(node)}
        if node.target in NO_DATA_FUNCTIONS:
            return None
    if node.op == "call_method" and node.target in NO_DATA_METHODS:
        return None
    raise NotImplementedError(f"{node.name}: {node.op} {node.target} has no kernel mapping")


#trace the model on one example input, one kernel per operator, deps from the graph's edges
def import_model(model, example, name, dtype_bytes=4):
    gm = symbolic_trace(model.eval())
    ShapeProp(gm).propagate(example)
    waits_for = {} #fx node -> the kernel ids a user of it has to wait for
    kernels = []
    for node in gm.graph.nodes:
        if node.op in ("placeholder", "get_attr", "output"):
            waits_for[node] = ()
            continue
        deps = tuple(dict.fromkeys(d for inp in node.all_input_nodes for d in waits_for[inp]))
        found = kernel_of(node, gm)
        if found is None:
            waits_for[node] = deps
            continue
        kind, shape = found
        kid = len(kernels)
        kernels.append(Kernel(id=kid, type=kind, shape=shape, deps=deps))
        waits_for[node] = (kid,)
    return Workload(name=name, source="torchfx", dtype_bytes=dtype_bytes, kernels=tuple(kernels))


if __name__ == "__main__":
    os.makedirs("data/workloads", exist_ok=True)
    w = import_model(torchvision.models.resnet50(weights=None), torch.randn(1, 3, 224, 224), "resnet50_b1")
    save(w, "data/workloads/resnet50_b1.json")
    print(f"{w.name}: {len(w.kernels)} kernels -> data/workloads/resnet50_b1.json")