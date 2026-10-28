"""Compare eager and compiled joint-loss parameter gradients on CUDA."""
import argparse
import copy
import json
from pathlib import Path
import torch
from gradformer.check import make_model
from gradformer.runtime import loss_forward, compile_model
from gradformer.core import joint_backward


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", required=True, choices=["nanochat", "llama2"])
    p.add_argument("--output", default="compile-correctness.json")
    args = p.parse_args()
    import wandb
    run = wandb.init(project="nanochat", entity="yibozhong657-none", job_type="correctness",
                     name=f"gradformer-{args.backend}-compile-correctness", config=vars(args))
    results = {}
    x = torch.arange(16, device="cuda").reshape(2, 8)
    y = (x + 1) % 128
    for mode in ("baseline", "sgd", "adam"):
        eager = make_model(args.backend, mode, "cuda")
        with torch.no_grad():
            for name, parameter in eager.named_parameters():
                if "c_proj.weight" in name:
                    parameter.normal_(0, .01)
        compiled = copy.deepcopy(eager)
        ntp, states, predictions = loss_forward(eager, args.backend, x, y, mode != "baseline")
        targets = torch.autograd.grad(ntp, states, retain_graph=True) if states else []
        loss = ntp.detach().clone()
        aux, _ = joint_backward(ntp, states, predictions, implementation="reference")
        live = compile_model(compiled, args.backend, mode, True)
        other, cs, cp = loss_forward(live, args.backend, x, y, mode != "baseline")
        ct = torch.autograd.grad(other, cs, retain_graph=True) if cs else []
        ca, _ = joint_backward(other, cs, cp, implementation="reference")
        torch.testing.assert_close(loss, other.detach(), rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(aux, ca, rtol=1e-4, atol=1e-5)
        for a, b in zip(targets, ct):
            torch.testing.assert_close(a, b, rtol=3e-3, atol=3e-5)
        maximum = 0.
        for (name, a), (other_name, b) in zip(eager.named_parameters(), compiled.named_parameters()):
            assert name == other_name
            assert a.grad is not None and b.grad is not None
            assert torch.isfinite(b.grad).all(), name
            torch.testing.assert_close(a.grad, b.grad, rtol=3e-3, atol=3e-5, msg=name)
            maximum = max(maximum, (a.grad - b.grad).abs().max().item())
        results[f"{mode}_max_abs_parameter_gradient_error"] = maximum
    results["passed"] = True
    Path(args.output).write_text(json.dumps(results, indent=2))
    run.summary.update(results)
    run.finish()


if __name__ == "__main__":
    main()
