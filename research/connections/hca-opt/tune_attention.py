"""Small deterministic tile sweep on the allocated debug GPU."""
import json
import os
from pathlib import Path
import torch
import triton
from nanochat import hca_triton as hca

assert os.environ['WANDB_MODE'] == 'disabled'
torch.manual_seed(443)
torch.set_num_threads(2)
out = Path('results/optimization'); out.mkdir(parents=True, exist_ok=True)
rows = []
for ratio in (16, 4):
    xs = [torch.randn(16, 2048, 4, 128, device='cuda', dtype=torch.bfloat16,
                      requires_grad=True) for _ in range(5)]
    starts = (torch.arange(2048, device='cuda') % ratio == 0).float()[None].expand(16, -1)
    dy = torch.randn_like(xs[0])
    hca.FIXED_FWD_TILE = hca.FIXED_BWD_TILE = (32, 32, 4)
    yref = hca.attention(*xs, starts, 256, ratio)
    gref = torch.autograd.grad(yref, xs, dy)
    for bm, bn, nw in [(32,32,4), (64,32,4), (32,64,4), (64,64,4),
                       (128,64,4), (64,128,4), (128,64,8), (128,128,8)]:
        hca.FIXED_FWD_TILE = hca.FIXED_BWD_TILE = (bm, bn, nw)
        def run():
            y = hca.attention(*xs, starts, 256, ratio)
            return y, torch.autograd.grad(y, xs, dy)
        y, gs = run()
        errors = []
        for a, e in zip((y, *gs), (yref, *gref)):
            assert torch.isfinite(a).all()
            rel = ((a.float()-e.float()).norm()/e.float().norm()).item()
            peak = ((a.float()-e.float()).abs().max()/e.float().abs().max()).item()
            assert rel < .02 and peak < .02, (bm,bn,nw,rel,peak)
            errors.append(dict(rel=rel, peak=peak))
        ms = triton.testing.do_bench(run, warmup=100, rep=300)
        row = dict(ratio=ratio, tile=[bm,bn,nw], milliseconds=ms, errors=errors,
                   gpu_uuid=str(torch.cuda.get_device_properties(0).uuid))
        print(json.dumps(row), flush=True)
        rows.append(row)
        (out/'tile-sweep.json').write_text(json.dumps(rows, indent=2)+'\n')
