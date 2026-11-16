"""Windowed Taylor statistics and physical FFN resizing for vanilla training.

Virtual gates are differentiable buffers, never optimizer parameters. The
single-GPU packed trainer owns the controller and its checkpoint metadata.
"""
import copy
import math
import random

import torch
from torch import nn
from torch.nn import functional as F

from nanochat.form_plan import form_budget, form_schedule, plan_reallocation


class VirtualGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, activation, gate):
        ctx.save_for_backward(activation, gate)
        return activation * gate.to(activation.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        activation, gate = ctx.saved_tensors
        # Keep the channel reduction in FP32 even with BF16 activations.
        dtype = torch.float64 if activation.dtype == torch.float64 else torch.float32
        gate_grad = (grad_output.to(dtype) * activation.to(dtype)).sum(tuple(range(activation.ndim - 1)))
        return grad_output * gate.to(grad_output.dtype), gate_grad.to(gate.dtype)


class FormMLP(nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.c_fc, self.c_proj = mlp.c_fc, mlp.c_proj
        device = self.c_fc.weight.device
        width = self.c_fc.out_features
        self.register_buffer('virtual_gate', torch.ones(width, device=device, requires_grad=True), persistent=False)
        self.register_buffer('form_mean', torch.zeros(width, device=device))
        self.register_buffer('form_second', torch.zeros(width, device=device))

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        # Moving a differentiable buffer creates a non-leaf without this detach.
        self.virtual_gate = self.virtual_gate.detach().requires_grad_(True)
        return self

    def forward(self, x):
        x = F.relu(self.c_fc(x)).square()
        if self.training and torch.is_grad_enabled():
            x = VirtualGate.apply(x, self.virtual_gate)
        return self.c_proj(x)

    def reset_statistics(self):
        width, device = self.c_fc.out_features, self.c_fc.weight.device
        self.virtual_gate = torch.ones(width, device=device, requires_grad=True)
        self.form_mean = torch.zeros(width, device=device, dtype=torch.float32)
        self.form_second = torch.zeros_like(self.form_mean)


def unpack_optimizer(model, optimizer):
    """Unstack Muon's group-owned slots into stable parameter names (one GPU)."""
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
        raise ValueError('Form currently uses the single-GPU packed training workflow')
    names = {p: name for name, p in model.named_parameters()}
    slots, settings = {}, {}
    for group in optimizer.param_groups:
        options = {k: v for k, v in group.items() if k != 'params'}
        if group['kind'] == 'muon':
            state = optimizer.state.get(group['params'][0], {})
            for i, p in enumerate(group['params']):
                slots[names[p]] = {k: v[i] for k, v in state.items()}
                settings[names[p]] = options.copy()
        else:
            for p in group['params']:
                slots[names[p]] = optimizer.state.get(p, {})
                settings[names[p]] = options.copy()
    return slots, settings


def resize_slot(tensor, keep, growth, axis):
    index = torch.tensor(keep, device=tensor.device, dtype=torch.long)
    retained = tensor.index_select(axis, index)
    shape = list(tensor.shape)
    shape[axis] = growth
    return torch.cat((retained, tensor.new_zeros(shape)), dim=axis)


def repack_optimizer(model, slots, settings):
    from nanochat.connection_model import optimizer_for
    optimizer, _ = optimizer_for(model)
    names = {p: name for name, p in model.named_parameters()}
    for group in optimizer.param_groups:
        group_settings = [settings[names[p]] for p in group['params']]
        # All matrices which are merged must share optimizer hyperparameters.
        assert all(s == group_settings[0] for s in group_settings)
        group.update(group_settings[0])
        if group['kind'] == 'muon':
            values = [slots[names[p]] for p in group['params']]
            if any(values):
                assert all(v.keys() == values[0].keys() for v in values)
                optimizer.state[group['params'][0]] = {
                    k: torch.stack([v[k] for v in values]) for k in values[0]}
        else:
            for p in group['params']:
                optimizer.state[p] = slots[names[p]]
    return optimizer


class FormController:
    def __init__(self, model, total_steps, seed=42):
        if model.config.form_mode == 'none':
            raise ValueError('Form is disabled')
        self.model = model
        c = model.config
        self.schedule = form_schedule(total_steps, c.form_count)
        self.total_width = 4 * c.width * c.depth
        self.budget = form_budget(self.total_width, c.depth, c.width, c.form_count)
        self.completed = self.window_steps = 0
        self.history = []
        self.seconds = 0.
        self.random = random.Random(seed + 17041)
        self.generator = torch.Generator(device=model.blocks[0].mlp.c_fc.weight.device)
        self.generator.manual_seed(seed + 17042)

    @property
    def mlps(self):
        return [b.mlp for b in self.model.blocks]

    def begin_step(self):
        for mlp in self.mlps:
            mlp.virtual_gate.grad = None

    @torch.no_grad()
    def observe(self):
        gradients = [m.virtual_gate.grad for m in self.mlps]
        if any(g is None for g in gradients):
            raise RuntimeError('Missing virtual gate gradients')
        if not bool(torch.stack([g.isfinite().all() for g in gradients]).all()):
            raise FloatingPointError('Nonfinite virtual gate gradient')
        beta = self.model.config.form_ema_decay
        for mlp, g in zip(self.mlps, gradients):
            mlp.form_mean.lerp_(g.float(), 1 - beta)
            mlp.form_second.lerp_(g.float().square(), 1 - beta)
        self.window_steps += 1

    def due(self, step):
        if self.completed == len(self.schedule):
            return False
        if step > self.schedule[self.completed]:
            raise RuntimeError('Missed Form boundary')
        return step == self.schedule[self.completed]

    def record_next_loss(self, step, loss):
        if self.history and self.history[-1]['step'] == step - 1:
            event = self.history[-1]
            event.update(next_train_ce=loss, next_batch_ce_delta=loss - event['preceding_train_ce'])
            return dict(event)
        return None

    @torch.no_grad()
    def apply(self, optimizer, step, preceding_loss):
        assert self.due(step)
        previous = self.schedule[self.completed - 1] if self.completed else 0
        assert self.window_steps == step - previous
        c = self.model.config
        scores = [(-m.form_mean + .5 * c.form_lambda * m.form_second).cpu().tolist() for m in self.mlps]
        plan = plan_reallocation(scores, c.width, self.budget,
                                 self.random if c.form_mode == 'random' else None)
        parameters_before = sum(p.numel() for p in self.model.parameters())
        slots, settings = unpack_optimizer(self.model, optimizer)
        for l, (mlp, keep, growth) in enumerate(zip(self.mlps, plan['keep'], plan['growth'])):
            if len(keep) != mlp.c_fc.out_features or growth:
                up = resize_slot(mlp.c_fc.weight, keep, growth, 0)
                down = resize_slot(mlp.c_proj.weight, keep, growth, 1)
                bound = .4 * math.sqrt(3 / c.width)
                if growth:
                    up[len(keep):].uniform_(-bound, bound, generator=self.generator)
                mlp.c_fc.weight = nn.Parameter(up)
                mlp.c_proj.weight = nn.Parameter(down)
                mlp.c_fc.out_features = mlp.c_proj.in_features = len(keep) + growth
                for suffix, axis in (('c_fc', 0), ('c_proj', 1)):
                    name = f'blocks.{l}.mlp.{suffix}.weight'
                    slots[name] = {k: resize_slot(v, keep, growth, axis) for k, v in slots[name].items()}
            mlp.reset_statistics()
        c.ffn_widths = plan['widths_after']
        assert sum(c.ffn_widths) == self.total_width and min(c.ffn_widths) >= c.width
        assert sum(p.numel() for p in self.model.parameters()) == parameters_before
        optimizer = repack_optimizer(self.model, slots, settings)
        # Drop graphs that captured old Parameters. Respecialize at next use;
        # account for that first step separately from steady-state throughput.
        if self.model.compiled_steps is not None:
            torch.compiler.reset()
            self.model.compiled_steps = [torch.compile(b.forward, dynamic=False, fullgraph=True)
                                         for b in self.model.blocks]
        self.completed += 1
        event = dict(number=self.completed, step=step, budget=self.budget,
                     window_steps=self.window_steps, widths_before=plan['widths_before'],
                     widths_after=plan['widths_after'], pruned=[len(p) for p in plan['prune']],
                     grown=plan['growth'], donor_counts=plan['donor_counts'],
                     parameters=parameters_before, preceding_train_ce=preceding_loss,
                     score_evaluation_forward_backward_passes=0,
                     loss_comparison='adjacent normal training batches; no extra evaluation probe',
                     scores=[dict(mean=sum(s) / len(s), minimum=min(s), maximum=max(s),
                                  pruned_sum=sum(s[j] for j in plan['prune'][l]),
                                  donor_sum=sum(s[j] for layer, j in plan['donors'] if layer == l))
                             for l, s in enumerate(scores)])
        self.history.append(event)
        self.window_steps = 0
        return optimizer, dict(event=copy.deepcopy(event), plan=plan, importance=scores)

    def state_dict(self):
        return dict(version=1, schedule=self.schedule, budget=self.budget, total_width=self.total_width,
                    completed=self.completed, window_steps=self.window_steps, history=copy.deepcopy(self.history),
                    seconds=self.seconds, random=self.random.getstate(), generator=self.generator.get_state())

    def load_state_dict(self, state, step):
        assert state['version'] == 1 and state['schedule'] == self.schedule
        assert state['budget'] == self.budget and state['total_width'] == self.total_width
        self.completed = state['completed']
        assert self.completed == sum(s <= step for s in self.schedule)
        self.window_steps = state['window_steps']
        assert self.window_steps == step - (self.schedule[self.completed - 1] if self.completed else 0)
        assert sum(self.model.config.ffn_widths) == self.total_width
        self.history = copy.deepcopy(state['history'])
        assert len(self.history) == self.completed
        self.seconds = state['seconds']
        self.random.setstate(state['random'])
        self.generator.set_state(state['generator'].cpu())
