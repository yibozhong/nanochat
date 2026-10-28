"""Native-backend construction and precision, shared between both forks."""
import math
import importlib
import os
import torch


def construct(backend, update, *, depth, width, heads, vocab, sequence, step_size, device):
    extra = dict(hidden_update=update, hidden_step_size=step_size,
                 hidden_beta1=.9, hidden_beta2=.999, hidden_eps=.01)
    if backend == "nanochat":
        from nanochat.gpt import GPT, GPTConfig
        model = GPT(GPTConfig(n_layer=depth, n_embd=width, n_head=heads, n_kv_head=heads,
                             vocab_size=vocab, sequence_len=sequence, window_pattern="L", **extra)).to(device)
        model.init_weights()
    else:
        from model import Transformer, ModelArgs
        model = Transformer(ModelArgs(n_layers=depth, dim=width, n_heads=heads, n_kv_heads=heads,
                                      vocab_size=vocab, max_seq_len=sequence, multiple_of=32,
                                      dropout=0.0, **extra)).to(device)
    return model


def optimizer_for(model, backend, batch_tokens):
    if backend == "nanochat":
        scale = math.sqrt(batch_tokens / 524288)
        return model.setup_optimizer(unembedding_lr=.008 * scale, embedding_lr=.3 * scale,
                                     matrix_lr=.02 * scale, scalar_lr=.5 * scale, weight_decay=.28)
    optimizer = model.configure_optimizers(.1, 5e-4, (.9, .95), "cuda")
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
    return optimizer


def loss_forward(model, backend, x, y, capture):
    extra = {"return_loss": True} if backend == "llama2" else {}
    output = model(x, y, capture_gradients=capture, **extra)
    if capture:
        value, states, predictions = output
    else:
        value, states, predictions = output, [], []
    return value, states, predictions


def compile_model(model, backend, update, enabled, compile_head=True):
    if not enabled:
        return model
    if update == "baseline":
        return torch.compile(model, dynamic=False)
    # Preserve h_l as explicit autograd nodes. Compiling the entire graph
    # makes returned intermediate states sibling outputs of one AOT function.
    # Blocks can be compiled without removing the gradient-target boundaries.
    blocks = model.transformer.h if backend == "nanochat" else model.layers
    # Adam bias correction specializes by depth; leave room for every layer
    # plus the SGD and zero-history specializations in this pinned PyTorch version.
    torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 2 * len(blocks) + 4)
    for block in blocks:
        block.forward = torch.compile(block.forward, dynamic=False)
    if compile_head:
        model._training_loss = torch.compile(model._training_loss, dynamic=False)
    native = importlib.import_module(model.__class__.__module__)
    if not hasattr(native.hidden_step, "_torchdynamo_orig_callable"):
        native.hidden_step = torch.compile(native.hidden_step, dynamic=False)
    import gradformer.core as core
    if not hasattr(core.prediction_loss, "_torchdynamo_orig_callable"):
        core.prediction_loss = torch.compile(core.prediction_loss, dynamic=False)
    return model
