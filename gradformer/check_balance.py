"""Independent gradient-reference checks for the single-GPU norm-balanced rule."""
import argparse
import copy
import json
from pathlib import Path

import torch

from gradformer.balance import NormBalancedBackward, for_model
from gradformer.check import make_model, forward
from gradformer.core import prediction_loss


def check(backend, device):
    results = {}
    x = torch.arange(16, device=device).reshape(2, 8)
    y = (x + 1) % 128
    for mode in ("sgd", "adam"):
        for accumulation in (1, 3):
            ref = make_model(backend, mode, device)
            with torch.no_grad():
                for name, parameter in ref.named_parameters():
                    if "c_proj.weight" in name:
                        parameter.normal_(0, .01)
            actual = copy.deepcopy(ref)
            balancer = for_model(actual, backend)
            parameters = list(ref.parameters())
            blocks = ref.transformer.h if backend == "nanochat" else ref.layers
            block_ids = {id(p) for block in blocks for p in block.parameters()}
            indices = [i for i, p in enumerate(parameters) if id(p) in block_ids]
            largest, ratios = 0., []
            for iteration in range(2):
                ref.zero_grad(set_to_none=True)
                actual.zero_grad(set_to_none=True)
                balancer.reset()
                ntp_losses, aux_losses = [], []
                for micro in range(accumulation):
                    offset = micro + iteration * 5
                    ntp, inputs, predictions = forward(ref, backend, x + offset, y + offset, True)
                    # Independent reference differentiates unscaled microbatch CE.
                    targets = torch.autograd.grad(ntp, inputs, retain_graph=True)
                    aux, _ = prediction_loss(predictions, targets, alpha=1., beta=1.)
                    ntp_losses.append(ntp / accumulation)
                    aux_losses.append(aux / accumulation)
                    an, ai, ap = forward(actual, backend, x + offset, y + offset, True)
                    balancer.backward(an, ai, ap, alpha=1., beta=1., accumulation=accumulation)
                nloss, aloss = sum(ntp_losses), sum(aux_losses)
                ng = torch.autograd.grad(nloss, parameters, retain_graph=True, allow_unused=True)
                ag = torch.autograd.grad(aloss, parameters, retain_graph=True, allow_unused=True)
                n = sum(g.square().sum() for i, g in enumerate(ng) if i in indices and g is not None).sqrt()
                a = sum(g.square().sum() for i, g in enumerate(ag) if i in indices and g is not None).sqrt()
                ratio = .5 if iteration == 0 else 1.
                coefficient = (ratio * n / a).detach()
                (nloss + coefficient * aloss).backward()
                diagnostic = balancer.finish(ratio=ratio, accumulation=accumulation)
                assert abs(diagnostic["balance/aux_to_ntp_ratio"] - ratio) < 2e-5
                assert abs(diagnostic["balance/weight"] / coefficient.item() - 1) < 2e-4
                assert diagnostic["balance/active"] == 1
                ratios.append(diagnostic["balance/aux_to_ntp_ratio"])
                for (name, p), (other, q) in zip(ref.named_parameters(), actual.named_parameters()):
                    assert name == other and p.grad is not None and q.grad is not None
                    torch.testing.assert_close(p.grad, q.grad, rtol=5e-4, atol=5e-6, msg=name)
                    largest = max(largest, (p.grad - q.grad).abs().max().item())
                    with torch.no_grad():
                        p.add_(p.grad, alpha=-.01)
                        q.add_(q.grad, alpha=-.01)
            results[f"{mode}_accumulation_{accumulation}"] = dict(max_abs_gradient_error=largest, ratios=ratios)
    # An auxiliary gradient of zero cannot be rescaled to a positive target.
    p = torch.nn.Parameter(torch.tensor([1., 2.], device=device))
    zero = NormBalancedBackward([p], [[p]])
    zero.ntp_gradients = [torch.ones_like(p)]
    zero.aux_gradients = [torch.zeros_like(p)]
    zero.microbatches = 1
    metric = zero.finish()
    assert metric["balance/active"] == 0 and metric["balance/weight"] == 0
    torch.testing.assert_close(p.grad, torch.ones_like(p))
    results["zero_auxiliary_guard"] = True
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["llama2", "nanochat"], required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import wandb
    run = wandb.init(project="nanochat", entity="yibozhong657-none", job_type="correctness",
                     name=f"gradformer-{args.backend}-gradient-balance-check", config=vars(args))
    try:
        results = dict(backend=args.backend, device=args.device, results=check(args.backend, args.device), passed=True)
        Path(args.output).write_text(json.dumps(results, indent=2))
        run.summary.update(results)
        print(json.dumps(results), flush=True)
    finally:
        run.finish()


if __name__ == "__main__":
    main()
