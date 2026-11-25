"""
torch.fx importer: an open model's real operator graph -> a workload
- torch.export traces the model down to aten operators (works for torchvision and hugging face models
  alike) and runs fake tensors through it, so every node knows its output shape
- conv2d -> gemm by im2col: m = batch x out h x out w, k = in channels x kernel h x kernel w, n = out channels
- linear -> gemm: m = all leading dims, k = in features, n = out features
- attention -> bgemm (q @ k^T, one per batch x head) -> softmax -> bgemm (p @ v)
- batchnorm, layernorm, relu, gelu, tanh, add, embedding lookup -> elementwise over the output
- pooling -> reduce over the input
- views, transposes, slices, dropout (eval) and integer / bool index and mask ops move no data: skipped,
  their users wait for what they waited for
anything else raises, so a new model never gets silently wrong kernels
run from the repo root: python -m workloads.torchfx
"""

import math
import os
import torch
import torchvision
from workloads.format import Kernel, Workload, save

aten = torch.ops.aten
ELEMENTWISE = {aten.batch_norm.default, aten.layer_norm.default, aten.relu.default, aten.relu_.default,
               aten.gelu.default, aten.tanh.default, aten.add.Tensor, aten.add_.Tensor,
               aten.embedding.default}
POOLS = {aten.max_pool2d.default, aten.adaptive_avg_pool2d.default}
NO_DATA = {aten.view.default, aten.reshape.default, aten.flatten.using_ints, aten.transpose.int,
           aten.unsqueeze.default, aten.expand.default, aten.slice.Tensor, aten.select.int,
           aten.contiguous.default, aten.dropout.default}
ATTENTION = aten.scaled_dot_product_attention.default


def shape(node):
    return tuple(node.meta["val"].shape)


def numel(node):
    return math.prod(shape(node))


#index and mask tensors (positions, token types, attention masks) are not compute
def is_index(node):
    return not node.meta["val"].dtype.is_floating_point


#one aten node -> (kernel type, shape), or None if it moves no data
def kernel_of(node):
    t = node.target
    if t in NO_DATA or is_index(node):
        return None
    if t == aten.conv2d.default:
        groups = node.args[6] if len(node.args) > 6 else 1
        if groups != 1:
            raise NotImplementedError(f"{node.name}: grouped conv")
        b, _, oh, ow = shape(node)
        out_ch, in_ch, kh, kw = shape(node.args[1]) #weight
        return "gemm", {"m": b * oh * ow, "n": out_ch, "k": in_ch * kh * kw}
    if t == aten.linear.default:
        out_f, in_f = shape(node.args[1]) #weight
        return "gemm", {"m": numel(node) // out_f, "n": out_f, "k": in_f}
    if t in ELEMENTWISE:
        return "elementwise", {"n": numel(node)}
    if t in POOLS:
        return "reduce", {"n": numel(node.args[0])}
    raise NotImplementedError(f"{node.name}: {t} has no kernel mapping")


#trace the model on one example input, one kernel per operator, deps from the graph's edges
def import_model(model, example, name, dtype_bytes=4):
    graph = torch.export.export(model.eval(), (example,)).graph_module.graph
    waits_for = {} #aten node -> the kernel ids a user of it has to wait for
    kernels = []

    def add(kind, shp, deps):
        kernels.append(Kernel(id=len(kernels), type=kind, shape=shp, deps=deps))
        return (len(kernels) - 1,)

    def deps_of(*nodes):
        return tuple(dict.fromkeys(d for n in nodes if isinstance(n, torch.fx.Node) for d in waits_for[n]))

    for node in graph.nodes:
        if node.op in ("placeholder", "get_attr", "output"):
            waits_for[node] = ()
            continue
        if node.target == ATTENTION: #q, k, v: (batch, heads, seq, head dim)
            q, k, v = node.args[:3]
            *bh, sq, d = shape(q)
            sk = shape(k)[-2]
            mask = node.kwargs.get("attn_mask", node.args[3] if len(node.args) > 3 else None)
            scores = add("bgemm", {"b": math.prod(bh), "m": sq, "n": sk, "k": d}, deps_of(q, k, mask))
            probs = add("elementwise", {"n": math.prod(bh) * sq * sk}, scores) #softmax
            waits_for[node] = add("bgemm", {"b": math.prod(bh), "m": sq, "n": d, "k": sk}, probs + deps_of(v))
            continue
        deps = deps_of(*node.all_input_nodes)
        found = kernel_of(node)
        waits_for[node] = deps if found is None else add(*found, deps)
    return Workload(name=name, source="torchfx", dtype_bytes=dtype_bytes, kernels=tuple(kernels))


def bert_base(seq_len=128):
    from transformers import BertConfig, BertModel #only needed for this model
    return BertModel(BertConfig()), torch.zeros(1, seq_len, dtype=torch.long) #bert-base, random weights


if __name__ == "__main__":
    os.makedirs("data/workloads", exist_ok=True)
    models = [("resnet50_b1", torchvision.models.resnet50(weights=None), torch.randn(1, 3, 224, 224)),
              ("bert_base_s128_b1", *bert_base(128))]
    for name, model, example in models:
        w = import_model(model, example, name)
        save(w, f"data/workloads/{name}.json")
        print(f"{w.name}: {len(w.kernels)} kernels -> data/workloads/{name}.json")