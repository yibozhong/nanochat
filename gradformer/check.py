"""CPU/GPU correctness checks. Run inside an allocated job, not on an access point."""
import argparse
import copy
import json
import os
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from gradformer.core import hidden_step, joint_backward, prediction_loss


def make_model(backend, mode, device):
    torch.manual_seed(123)
    kw = dict(hidden_update=mode, hidden_step_size=1.0 if mode == "sgd" else 0.01)
    if backend == "nanochat":
        from nanochat.gpt import GPT, GPTConfig
        config = GPTConfig(sequence_len=16, vocab_size=128, n_layer=2, n_head=2,
                           n_kv_head=2, n_embd=32, window_pattern="L", **kw)
        model = GPT(config).to(device)
        model.init_weights()
    else:
        from model import Transformer, ModelArgs
        config = ModelArgs(max_seq_len=16, vocab_size=128, n_layers=2, n_heads=2,
                           n_kv_heads=2, dim=32, multiple_of=16, **kw)
        model = Transformer(config).to(device)
    return model


def forward(model, backend, x, y, capture):
    result = model(x, y, capture_gradients=capture)
    if capture:
        output, states, preds = result
    else:
        output, states, preds = result, [], []
    loss = output if backend == "nanochat" else model.last_loss
    return loss, states, preds


def check(backend, device):
    results = {}
    x = torch.arange(16, device=device).reshape(2, 8)
    y = (x + 1) % 128
    baseline = make_model(backend, "baseline", device)
    sgd = make_model(backend, "sgd", device)
    sgd.load_state_dict(baseline.state_dict(), strict=False)
    a, _, _ = forward(baseline, backend, x, y, False)
    b, _, _ = forward(sgd, backend, x, y, True)
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    a.backward()
    b.backward()
    for name, p in baseline.named_parameters():
        torch.testing.assert_close(p.grad, dict(sgd.named_parameters())[name].grad, rtol=2e-4, atol=2e-6)
    results["baseline_sgd_identity"] = True
    for mode in ("sgd", "adam"):
        ref = make_model(backend, mode, device)
        # Nonzero residuals also exercise Adam's depth history and preceding losses.
        with torch.no_grad():
            for name, p in ref.named_parameters():
                if "c_proj.weight" in name:
                    p.normal_(0, 0.01)
        opt = copy.deepcopy(ref)
        for model, implementation in ((ref, "reference"), (opt, "reuse")):
            model.zero_grad(set_to_none=True)
            for offset in (0, 3):
                ntp, inputs, preds = forward(model, backend, x + offset, y + offset, True)
                aux, metrics = joint_backward(ntp, inputs, preds, weight=0.1,
                                             accumulation=2, implementation=implementation)
                assert torch.isfinite(aux) and torch.isfinite(metrics).all()
        largest = 0.0
        for (name, p), (other_name, q) in zip(ref.named_parameters(), opt.named_parameters()):
            assert name == other_name and p.grad is not None and q.grad is not None
            assert torch.isfinite(p.grad).all() and torch.isfinite(q.grad).all(), name
            torch.testing.assert_close(p.grad, q.grad, rtol=3e-4, atol=3e-6, msg=name)
            largest = max(largest, (p.grad - q.grad).abs().max().item())
        results[mode + "_reference_reuse_max_abs"] = largest
        # Targets are constants under differentiation.
        pred = torch.randn(2, 4, 8, device=device, requires_grad=True)
        target = torch.randn_like(pred, requires_grad=True)
        aux, _ = prediction_loss([pred], [target])
        gp, gt = torch.autograd.grad(aux, (pred, target), allow_unused=True)
        assert gp is not None and gt is None
        # No label inference, reset of moments, and prefix causality.
        ref.eval()
        with torch.no_grad():
            if backend == "nanochat":
                out1 = ref(x)
                changed = x.clone(); changed[:, 4:] += 17
                out2 = ref(changed)
                torch.testing.assert_close(out1[:, :4], out2[:, :4], atol=1e-6, rtol=1e-5)
                torch.testing.assert_close(ref(x), out1, atol=0, rtol=0)
            else:
                out1 = ref(x[:, :4])
                out2 = ref(x[:, :4])
                torch.testing.assert_close(out1, out2, atol=0, rtol=0)
                # Training path returns all logits, even without a backward.
                full1 = ref(x, y)
                changed = x.clone(); changed[:, 4:] += 17
                full2 = ref(changed, y)
                torch.testing.assert_close(full1[:, :4], full2[:, :4], atol=1e-6, rtol=1e-5)
        results[mode + "_causal_and_inference"] = True
    # Exact differentiable Adam versus finite differences, including zero prediction.
    cfg = SimpleNamespace(hidden_update="adam", hidden_beta1=.9, hidden_beta2=.999, hidden_eps=.01)
    h = torch.randn(2, 3, dtype=torch.double, device=device, requires_grad=True)
    g = torch.randn_like(h, requires_grad=True)
    raw = torch.tensor(-2., device=device, dtype=torch.double, requires_grad=True)
    fun = lambda a, b, c: hidden_step(a, b, c, None, 0, cfg)[0]
    assert torch.autograd.gradcheck(fun, (h, g, raw), fast_mode=True)
    zero = torch.zeros_like(g, requires_grad=True)
    assert torch.autograd.gradcheck(fun, (h, zero, raw), eps=1e-8, atol=1e-5, rtol=1e-4, fast_mode=True)
    # Upstream nanochat's exact zero residual initialization must not yield NaNs.
    initial = make_model(backend, "adam", device)
    ntp, inputs, preds = forward(initial, backend, x, y, True)
    joint_backward(ntp, inputs, preds)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in initial.parameters())
    results["adam_gradcheck_and_zero_initialization"] = True
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["nanochat", "llama2"], required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="correctness.json")
    args = parser.parse_args()
    import wandb
    run = wandb.init(project="nanochat", entity="yibozhong657-none",
                     job_type="correctness", config=vars(args),
                     name="gradformer-" + args.backend + "-correctness")
    try:
        results = check(args.backend, args.device)
        results.update(backend=args.backend, torch_version=torch.__version__, passed=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        run.summary.update(results)
        print(json.dumps(results, indent=2))
    finally:
        run.finish()
