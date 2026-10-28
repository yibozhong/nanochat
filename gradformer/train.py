"""Matched-budget native-backend pretraining with the original joint objective.

Run with uv run --extra gpu --group gradformer python -m gradformer.train ...
or torchrun on a single multi-GPU node. W&B logging is mandatory.
"""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist

from gradformer.core import joint_backward
from gradformer.runtime import construct, optimizer_for, loss_forward, compile_model


def atomic_save(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def batch(data, rng, batch_size, sequence, device):
    starts = rng.integers(0, len(data) - sequence - 1, size=batch_size)
    xy = np.stack([data[s:s + sequence + 1] for s in starts]).astype(np.int64)
    xy = torch.from_numpy(xy).to(device)
    return xy[:, :-1].contiguous(), xy[:, 1:].contiguous()


def main():
    run_started = time.perf_counter()
    wall_before = 0.0
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["nanochat", "llama2"], required=True)
    p.add_argument("--update", choices=["baseline", "sgd", "adam"], required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--sequence", type=int, default=256)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--batch-tokens", type=int, default=65536)
    p.add_argument("--tokens", type=int, default=200_000_000)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--aux-weight", type=float, default=.1)
    p.add_argument("--alpha", type=float, default=1.)
    p.add_argument("--beta", type=float, default=.1)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-batches", type=int, default=16)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--backward", choices=["reference", "reuse"], default="reference")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--stop-after-step", type=int, default=0, help="Controlled interruption; keep original schedule horizon")
    p.add_argument("--variant", default=None, help="Display label for architecture/supervision ablation")
    p.add_argument("--phase", choices=["smoke", "pretrain"], default="pretrain")
    args = p.parse_args()
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    if world > 1:
        dist.init_process_group("nccl", device_id=device)
    assert args.batch_tokens % (args.batch * args.sequence * world) == 0
    accumulation = args.batch_tokens // (args.batch * args.sequence * world)
    steps = math.ceil(args.tokens / args.batch_tokens)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((Path(args.data) / "manifest.json").read_text())
    train = np.memmap(Path(args.data) / "train.bin", dtype=np.uint16, mode="r")
    val = np.memmap(Path(args.data) / "val.bin", dtype=np.uint16, mode="r")
    rng = np.random.default_rng(args.seed + 100003 * rank)
    torch.manual_seed(args.seed)
    model = construct(args.backend, args.update, depth=args.depth, width=args.width, heads=args.heads,
                      vocab=manifest["vocab_size"], sequence=args.sequence,
                      step_size=1.0 if args.update == "sgd" else .01, device=device)
    optimizer = optimizer_for(model, args.backend, args.batch_tokens)
    source_path = Path("gradformer/source.json")
    source = json.loads(source_path.read_text()) if source_path.exists() else {"status": "unversioned"}
    configuration = {**vars(args), "world_size": world, "accumulation": accumulation,
                     "actual_tokens": steps * args.batch_tokens, "dataset": manifest, "source": source,
                     "model": asdict(model.config if args.backend == "nanochat" else model.params),
                     "parameters": sum(t.numel() for t in model.parameters()),
                     "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                     "target": "unscaled per-microbatch mean NTP input gradient",
                     "backward": args.backward, "auxiliary_scope": "full connected graph",
                     "optimizer_groups": [{k: v for k, v in g.items() if k != "params"}
                                          for g in optimizer.param_groups]}
    start_step, training_seconds, best = 0, 0., float("inf")
    latest_path = out / "latest.json"
    if args.resume and latest_path.exists():
        latest = json.loads(latest_path.read_text())
        directory = out / latest["directory"]
        model.load_state_dict(torch.load(directory / "model.pt", map_location=device, weights_only=False))
        state = torch.load(directory / f"rank-{rank}.pt", map_location=device, weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        rng.bit_generator.state = state["data_rng"]
        torch.set_rng_state(state["torch_rng"].cpu())
        torch.cuda.set_rng_state(state["cuda_rng"].cpu())
        start_step, training_seconds, best = state["step"], state["training_seconds"], state["best_val"]
        wall_before = state.get("wall_seconds", 0.0)
    live = compile_model(model, args.backend, args.update, args.compile)
    import wandb
    run = None
    if rank == 0:
        id_path = out / "wandb-id.txt"
        run_id = id_path.read_text().strip() if args.resume and id_path.exists() else wandb.util.generate_id()
        id_path.write_text(run_id)
        run = wandb.init(project="nanochat", entity="yibozhong657-none", id=run_id, resume="allow",
                         name=f"gradformer-{args.backend}-{args.variant or args.update}-seed{args.seed}-{args.phase}",
                         group=f"gradformer-{args.backend}-{args.tokens}tokens",
                         tags=[args.backend, args.update, "joint-ntp-raw-gradient"],
                         job_type=args.phase, config=configuration)
        (out / "config.json").write_text(json.dumps(configuration, indent=2))
    log_file = open(out / "metrics.jsonl", "a") if rank == 0 else None

    def emit(record):
        if rank == 0:
            record["wall_seconds"] = wall_before + time.perf_counter() - run_started
            log_file.write(json.dumps(record) + "\n"); log_file.flush()
            run.log(record, step=record["step"])
            print(json.dumps(record), flush=True)

    def context():
        return torch.autocast("cuda", dtype=torch.bfloat16) if args.backend == "llama2" else nullcontext()

    def evaluate():
        model.eval()
        vrng = np.random.default_rng(987654 + rank)
        losses = []
        with torch.no_grad():
            for _ in range(args.eval_batches):
                vx, vy = batch(val, vrng, args.batch, args.sequence, device)
                with context():
                    vl, _, _ = loss_forward(model, args.backend, vx, vy, False)
                losses.append(vl.detach())
        value = torch.stack(losses).mean()
        if world > 1:
            dist.all_reduce(value, op=dist.ReduceOp.AVG)
        model.train()
        return value.item()

    def save(step):
        # Two alternating generations; latest.json changes only after all shards finish.
        previous = json.loads(latest_path.read_text())["directory"] if latest_path.exists() else "checkpoint-1"
        directory = out / ("checkpoint-1" if previous == "checkpoint-0" else "checkpoint-0")
        directory.mkdir(exist_ok=True)
        atomic_save(dict(optimizer=optimizer.state_dict(), data_rng=rng.bit_generator.state,
                         torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                         step=step, training_seconds=training_seconds, best_val=best,
                         wall_seconds=wall_before + time.perf_counter() - run_started),
                    directory / f"rank-{rank}.pt")
        if rank == 0:
            atomic_save(model.state_dict(), directory / "model.pt")
        if world > 1:
            dist.barrier()
        if rank == 0:
            temporary = out / "latest.json.tmp"
            temporary.write_text(json.dumps(dict(directory=directory.name, step=step)))
            temporary.replace(latest_path)
        if world > 1:
            dist.barrier()

    value = evaluate()
    if value < best:
        best = value
        if rank == 0:
            atomic_save(model.state_dict(), out / "best-model.pt")
    emit(dict(step=start_step, tokens=start_step * args.batch_tokens, val_ce=value,
              training_seconds=training_seconds))
    torch.cuda.reset_peak_memory_stats()
    for step in range(start_step, steps):
        warmup = min(100, max(1, int(.05 * steps)))
        if step < warmup:
            multiplier = (step + 1) / warmup
        elif args.backend == "nanochat":
            left = (steps - step) / max(1, round(.65 * steps))
            multiplier = .05 + .95 * min(1., left)
        else:
            multiplier = .5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup)))
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * multiplier
            if group.get("kind") == "muon":
                group["momentum"] = .85 + .12 * min(1., step / 400)
                group["weight_decay"] = .28 * .5 * (1 + math.cos(math.pi * step / steps))
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); begin = time.perf_counter()
        micro_losses, micro_aux = [], []
        metrics = None
        for _ in range(accumulation):
            x, y = batch(train, rng, args.batch, args.sequence, device)
            with context():
                ntp, states, preds = loss_forward(live, args.backend, x, y, args.update != "baseline")
            aux, metrics = joint_backward(ntp, states, preds, weight=args.aux_weight,
                                          alpha=args.alpha, beta=args.beta, accumulation=accumulation,
                                          implementation=args.backward)
            if not torch.isfinite(aux).item():
                raise RuntimeError(f"Nonfinite auxiliary loss at step {step}")
            micro_losses.append(ntp.detach()); micro_aux.append(aux)
        if args.backend == "llama2":
            if world > 1:
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        dist.all_reduce(parameter.grad, op=dist.ReduceOp.AVG)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        torch.cuda.synchronize()
        duration = time.perf_counter() - begin
        training_seconds += duration
        loss = torch.stack(micro_losses).mean()
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"Nonfinite loss at step {step}")
        if (step + 1) % 10 == 0 or step == start_step:
            values = torch.stack((loss, torch.stack(micro_aux).mean()))
            if world > 1:
                dist.all_reduce(values, op=dist.ReduceOp.AVG)
            record = dict(step=step + 1, tokens=(step + 1) * args.batch_tokens,
                          train_ce=values[0].item(), gradient_loss=values[1].item(),
                          step_seconds=duration, tokens_per_second=args.batch_tokens / duration,
                          training_seconds=training_seconds,
                          peak_memory_gb=torch.cuda.max_memory_allocated() / 2**30)
            if metrics is not None:
                if world > 1:
                    dist.all_reduce(metrics, op=dist.ReduceOp.AVG)
                for layer, row in enumerate(metrics.cpu().tolist()):
                    for key, value in zip(("mse", "cosine", "pred_norm", "target_norm", "valid_cosine"), row):
                        record[f"layer/{layer}/{key}"] = value
                for layer, value in enumerate(torch.nn.functional.softplus(model.hidden_step_raw).detach().cpu().tolist()):
                    record[f"layer/{layer}/step_size"] = value
            emit(record)
        if (step + 1) % args.eval_every == 0 or step + 1 == steps:
            value = evaluate()
            if value < best:
                best = value
                if rank == 0:
                    atomic_save(model.state_dict(), out / "best-model.pt")
            emit(dict(step=step + 1, tokens=(step + 1) * args.batch_tokens, val_ce=value,
                      best_val_ce=best, training_seconds=training_seconds))
        if (step + 1) % args.save_every == 0 or step + 1 == steps or step + 1 == args.stop_after_step:
            save(step + 1)
        if step + 1 == args.stop_after_step and step + 1 < steps:
            if rank == 0:
                run.summary.update(dict(completed=False, stopped_for_resume=True, last_step=step + 1))
                run.finish(); log_file.close()
            if world > 1:
                dist.destroy_process_group()
            return
    if rank == 0:
        wall_seconds = wall_before + time.perf_counter() - run_started
        run.summary.update(dict(best_val_ce=best, final_val_ce=value, training_seconds=training_seconds,
                                wall_seconds=wall_seconds,
                                tokens=steps * args.batch_tokens, completed=True))
        (out / "completed.json").write_text(json.dumps(dict(best_val_ce=best, final_val_ce=value,
                                                           training_seconds=training_seconds, wall_seconds=wall_seconds,
                                                           tokens=steps * args.batch_tokens)))
        run.finish(); log_file.close()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
