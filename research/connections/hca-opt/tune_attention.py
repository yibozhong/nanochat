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
    hca.FIXED_FWD_TILE = hca.FIXED_BWD_TILE = hca.FIXED_SMALL_TILE = (32, 32, 4)
    hca.FIXED_LOCAL_DKV_TILE = hca.FIXED_SUMMARY_TILE = (32,32,4)
    hca.FIXED_DKV_SPLITS = hca.FIXED_SMALL_SPLITS = 1
    hca.FIXED_BWD_STAGES = 3
    yref = hca.attention(*xs, starts, 256, ratio)
    gref = torch.autograd.grad(yref, xs, dy)
    for bm, bn, nw, stages in [(32,32,4,3), (64,32,4,3), (32,64,4,3), (64,64,4,3),
                       (128,64,4,1), (64,128,4,1), (128,64,8,1), (128,128,8,1),
                       (64,64,4,1), (128,64,4,2)]:
        hca.FIXED_BWD_STAGES = stages
        hca.FIXED_FWD_TILE = hca.FIXED_BWD_TILE = hca.FIXED_SMALL_TILE = (bm, bn, nw)
        hca.FIXED_LOCAL_DKV_TILE = hca.FIXED_SUMMARY_TILE = (bm,bn,nw)
        def run():
            y = hca.attention(*xs, starts, 256, ratio)
            return y, torch.autograd.grad(y, xs, dy)
        try:
            y, gs = run()
        except triton.OutOfResources as e:
            rows.append(dict(ratio=ratio,tile=[bm,bn,nw],stages=stages,status='out_of_resources',reason=str(e)))
            (out/'tile-sweep.json').write_text(json.dumps(rows,indent=2)+'\n')
            continue
        errors = []
        for a, e in zip((y, *gs), (yref, *gref)):
            assert torch.isfinite(a).all()
            rel = ((a.float()-e.float()).norm()/e.float().norm()).item()
            peak = ((a.float()-e.float()).abs().max()/e.float().abs().max()).item()
            assert rel < .02 and peak < .02, (bm,bn,nw,rel,peak)
            errors.append(dict(rel=rel, peak=peak))
        ms = triton.testing.do_bench(run, warmup=100, rep=300)
        row = dict(ratio=ratio, tile=[bm,bn,nw], stages=stages, milliseconds=ms, errors=errors,
                   gpu_uuid=str(torch.cuda.get_device_properties(0).uuid))
        print(json.dumps(row), flush=True)
        rows.append(row)
        (out/'tile-sweep.json').write_text(json.dumps(rows, indent=2)+'\n')
