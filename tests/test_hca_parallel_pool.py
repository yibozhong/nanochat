"""Parallel carries must preserve the original hard recurrence, including tails."""
import pytest
import torch
import importlib.util
from pathlib import Path
from nanochat.hca_triton import prefix_pool

_path=Path(__file__).resolve().parents[1]/'research/connections/hca-opt/parallel_pool.py'
_spec=importlib.util.spec_from_file_location('parallel_pool_candidate',_path)
_candidate=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_candidate)
hard_prefix_pool=_candidate.hard_prefix_pool


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
@pytest.mark.parametrize('kind',['single','every','mixed'])
@pytest.mark.parametrize('length',[1,63,64,65,137,2048])
def test_parallel_hard_pool(dtype,kind,length):
    torch.manual_seed(888)
    b,h,d=2,2,32
    xs=[torch.randn(b,length,h,d,device='cuda',dtype=dt,requires_grad=True)
        for dt in (dtype,dtype,torch.float32)]
    starts=(torch.rand(b,length,device='cuda')<.07).float()
    if kind=='single':starts.zero_()
    if kind=='every':starts.fill_(1)
    starts[:,0]=1
    actual=hard_prefix_pool(*xs,starts);expected=prefix_pool(*xs,starts)
    for a,e in zip(actual,expected):
        torch.testing.assert_close(a,e,rtol=0,atol=0)
    gs=[torch.randn_like(x) for x in actual]
    ga=torch.autograd.grad(actual,xs,gs,retain_graph=True)
    ge=torch.autograd.grad(expected,xs,gs)
    for a,e in zip(ga,ge):
        # The identical arithmetic is deliberate, not merely similar VJPs.
        torch.testing.assert_close(a,e,rtol=0,atol=0)
    repeated=torch.autograd.grad(actual,xs,gs)
    for a,e in zip(ga,repeated):torch.testing.assert_close(a,e,rtol=0,atol=0)
