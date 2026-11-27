"""First-order, reverse-order block updates for the single-GPU vanilla trainer."""
from contextlib import contextmanager

import torch

from nanochat.gpt import norm


def split_optimizer_groups(model, groups):
    """Keep the baseline hyperparameters and give each reverse step its own Muon buffers."""
    from nanochat.connection_model import ConnectionModel

    if type(model) is not ConnectionModel or model.config.variant != 'baseline':
        raise ValueError('Sequential backward requires the vanilla baseline with independent blocks')
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
        raise ValueError('The vanilla sequential trainer supports a single GPU')
    owners = {}
    storage = {}
    # Head and embedding are ordinary backprop endpoints and retain their distinct AdamW groups.
    for stage, module in [*enumerate(model.blocks), ('endpoints', model.wte), ('endpoints', model.lm_head)]:
        for weight in module.parameters():
            if not weight.requires_grad:
                raise ValueError('Sequential vanilla training expects trainable parameters in every stage')
            address = (weight.device, weight.untyped_storage().data_ptr())
            if address in storage:
                raise ValueError(f'Sequential backward cannot independently update shared parameters in {storage[address]} and {stage}')
            owners[id(weight)] = stage
            storage[address] = stage
    if set(owners) != {id(weight) for weight in model.parameters()}:
        raise ValueError('A trainable parameter lies outside the sequential blocks and endpoints')
    result = []
    for group in groups:
        by_stage = {}
        for weight in group['params']:
            by_stage.setdefault(owners[id(weight)], []).append(weight)
        start = 0
        for stage, params in by_stage.items():
            owned = dict(group, params=params, sequential_stage=stage)
            if group['kind'] == 'muon':
                # Preserve the original BF16 fused-kernel batch geometry and matrix slots.
                assert all(p is q for p, q in zip(params, group['params'][start:start + len(params)], strict=True))
                owned['sequential_layout'] = (start, len(group['params']))
            result.append(owned)
            start += len(params)
    return result


@contextmanager
def _input_grad_only(params):
    # All original graphs for this block have been consumed before entry.
    # AOTAutograd can now compile a VJP that only computes the input derivative.
    try:
        for p in params:
            p.requires_grad_(False)
        yield
    finally:
        for p in params:
            p.requires_grad_(True)


def _check_finite(tensors):
    if not torch.stack([torch.isfinite(t.detach()).all() for t in tensors]).all().item():
        raise FloatingPointError('Nonfinite sequential loss or gradient; abort this step and recover from a complete checkpoint')


def sequential_step(model, opt, data, step, *, microbatch=16, batch_tokens=32768):
    """Accumulate a complete effective batch before each block's single update.

    The original forward has disconnected block graphs. With accumulation, keep
    only boundary activations and rematerialize each parameter VJP one microbatch
    at a time. Checkpoints and schedule steps are outside this function.
    """
    length = model.config.sequence_length
    if batch_tokens <= 0 or microbatch <= 0 or batch_tokens % (microbatch * length):
        raise ValueError('The effective batch must be a positive whole number of microbatches')
    rows = batch_tokens // length
    batches = rows // microbatch
    remat = batches > 1 or model.config.activation_checkpoint
    opt.zero_grad(set_to_none=True)
    functions = model.compiled_steps or model.blocks
    cos, sin = model.cos[:, :length], model.sin[:, :length]
    inputs = [[] for _ in model.blocks]
    original_outputs = []
    embeddings, incoming, losses = [], [], []

    for offset in range(0, rows, microbatch):
        tokens, targets = data.batch('train', step * rows + offset, microbatch)
        embedding = norm(model.wte(tokens))
        embeddings.append(embedding)
        h = embedding.detach()
        for i, fn in enumerate(functions):
            inputs[i].append(h)
            with torch.set_grad_enabled(not remat):
                output = fn(h, cos, sin, None)
            if not remat:
                original_outputs.append(output)
            h = output.detach()
        top = h.requires_grad_(True)
        loss = model.readout(norm(top), targets) / batches
        loss.backward()
        losses.append(loss.detach())
        incoming.append(top.grad.detach())
    total = torch.stack(losses).sum()
    _check_finite([total, model.lm_head.weight.grad, *incoming])

    for i in reversed(range(len(model.blocks))):
        fn = functions[i]
        params = tuple(model.blocks[i].parameters())
        for hidden, upstream in zip(inputs[i], incoming):
            out = fn(hidden, cos, sin, None) if remat else original_outputs.pop()
            out.backward(upstream)
        _check_finite([p.grad for p in params])
        opt.step(stage=i)
        with _input_grad_only(params):
            for j, hidden in enumerate(inputs[i]):
                leaf = hidden.detach().requires_grad_(True)
                updated = fn(leaf, cos, sin, None)
                (dx,) = torch.autograd.grad(updated, leaf, incoming[j], create_graph=False)
                incoming[j] = dx.detach()
        inputs[i].clear()
    for embedding, upstream in zip(embeddings, incoming):
        embedding.backward(upstream)
    _check_finite([model.wte.weight.grad, *incoming])
    opt.step(stage='endpoints')
    return total.item()


def sequential_cost(config, batch, microbatch):
    """Algorithmic matrix FLOPs on the same causal-pair convention as the baseline."""
    from nanochat.connection_cost import model_cost

    cost = model_cost(config, batch=batch)
    parts = cost['components_forward']
    projections, attention = parts['backbone_projections'], parts['attention']
    # Two block forwards. The old parameter VJP skips input gradients for
    # the three first Q/K/V projections. The new VJP computes input gradients
    # through all linears. Both VJPs differentiate the full attention product.
    qkv = 2 * batch * config.sequence_length * config.depth * 3 * config.width**2
    extra = 2 * projections - qkv + 3 * attention
    if microbatch < batch or config.activation_checkpoint:
        extra += projections + attention
    cost['standard_training_flops'] = cost['estimated_training_flops']
    cost['extra_block_flops'] = extra
    cost['estimated_training_flops'] += extra
    cost['accounting'] = ('Causal matrix FLOPs including updated block forward/input VJP and any old-forward '
                          'rematerialization. Excludes optimizer, norms, nonlinearities and kernel-internal recomputation.')
    return cost
