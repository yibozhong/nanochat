"""Measure full training steps on matched native backends and allocated GPUs."""
import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
from gradformer.core import joint_backward
from gradformer.runtime import construct, optimizer_for, loss_forward, compile_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["nanochat", "llama2"], required=True)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--vocab", type=int, default=32000)
    parser.add_argument("--sequence", type=int, default=256)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--no-compile-head", action="store_true", help="Ablate vocabulary-loss fusion")
    parser.add_argument("--output", default="benchmark.json")
    args = parser.parse_args()
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group("nccl", device_id=torch.device("cuda", local))
    device = torch.device("cuda", local)
    source_path = Path("gradformer/source.json")
    source = json.loads(source_path.read_text()) if source_path.exists() else {"status": "unversioned"}
    import wandb
    run = wandb.init(project="nanochat", entity="yibozhong657-none", job_type="benchmark",
                     name=f"gradformer-{args.backend}-benchmark-{'compile' if args.compile else 'eager'}",
                     config={**vars(args), "world_size": world, "source": source, "seed": 1337,
                             "torch": torch.__version__,
                             "gpu": torch.cuda.get_device_name(local)}) if rank == 0 else None
    results = []
    cases = [("baseline", "reuse"), ("sgd", "reuse"), ("adam", "reference"), ("adam", "reuse")]
    for update, implementation in cases:
        torch.manual_seed(1337)
        model = construct(args.backend, update, depth=args.depth, width=args.width, heads=args.heads,
                          vocab=args.vocab, sequence=args.sequence, step_size=1.0 if update == "sgd" else .01,
                          device=device)
        optimizer = optimizer_for(model, args.backend, args.batch * args.sequence * world)
        live = compile_model(model, args.backend, update, args.compile, compile_head=not args.no_compile_head)
        rng = torch.Generator(device=device).manual_seed(42 + rank)
        x = torch.randint(args.vocab, (args.batch, args.sequence), device=device, generator=rng)
        y = torch.randint(args.vocab, x.shape, device=device, generator=rng)
        times = []
        if world > 1:
            dist.barrier()
        for step in range(args.warmup + args.steps):
            optimizer.zero_grad(set_to_none=True)
            if step == args.warmup:
                torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            ctx = torch.autocast("cuda", dtype=torch.bfloat16) if args.backend == "llama2" else nullcontext()
            with ctx:
                ntp, states, preds = loss_forward(live, args.backend, x, y, update != "baseline")
            aux, metrics = joint_backward(ntp, states, preds, weight=.1, implementation=implementation)
            if args.backend == "llama2" and world > 1:
                for p in model.parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad)
                        p.grad.div_(world)
            if args.backend == "llama2":
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if step >= args.warmup:
                times.append(elapsed)
            if not torch.isfinite(ntp).item() or not torch.isfinite(aux).item():
                raise RuntimeError(f"Nonfinite loss: {update}/{implementation}")
        peak = torch.tensor([torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved(),
                             statistics.median(times)], dtype=torch.float64, device=device)
        if world > 1:
            dist.all_reduce(peak, op=dist.ReduceOp.MAX)
        row = dict(update=update, backward=implementation, compiled=args.compile,
                   step_seconds=peak[2].item(),
                   tokens_per_second=args.batch * args.sequence * world / peak[2].item(),
                   peak_allocated_gb=peak[0].item() / 2**30, peak_reserved_gb=peak[1].item() / 2**30,
                   parameters=sum(p.numel() for p in model.parameters()),
                   final_ntp=ntp.item(), final_aux=aux.item(), timings=times)
        results.append(row)
        if run:
            run.log({k: v for k, v in row.items() if k != "timings"})
            print(json.dumps(row), flush=True)
        del live, model, optimizer, ntp, states, preds, aux, metrics
        gc.collect()
        torch.cuda.empty_cache()
    if rank == 0:
        for row in results:
            row["slowdown_vs_baseline"] = row["step_seconds"] / results[0]["step_seconds"]
        payload = dict(config=vars(args), source=source, world_size=world, gpu=torch.cuda.get_device_name(),
                       torch=torch.__version__, results=results)
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2)
        run.summary["results"] = results
        run.finish()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
