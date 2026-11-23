"""GPU checks for the zero-initialized additive read."""
from dataclasses import replace
import pytest
import torch

from nanochat.attention_residuals import reference_mix
from nanochat.chord_triton import bf16_query, corrected_query
from nanochat.connection_model import optimizer_for
from nanochat.engine import KVCache
from nanochat.gpt import norm
from nanochat.packed_training import seed_all
from test_chord_attnres import setting
from test_momentum import gpu, inputs, make, test_exact_optimizer_checkpoint_next_update as check_resume


@pytest.mark.parametrize('d,g', [(128, 0.), (512, -.45), (512, .3)])
def test_corrected_query_and_gradient(d,g):
    h, v, memory = [torch.randn(3, 17, d, device='cuda', dtype=torch.bfloat16).requires_grad_() for _ in range(3)]
    mu = torch.tensor([.4], device='cuda', requires_grad=True)
    gain = torch.tensor([g], device='cuda', requires_grad=True)
    query = bf16_query(h,v,mu,False)
    ref = norm((query.float()+gain*memory.float()).to(query.dtype))
    out = corrected_query(h,v,mu,memory,gain)
    torch.testing.assert_close(out,ref,rtol=.008,atol=.008)
    if g == 0:
        assert torch.equal(out, bf16_query(h,v,mu,False,normalize=True))
    targets = (h,v,mu,memory,gain)
    dy = torch.randn_like(out)
    for ideal, got in zip(torch.autograd.grad(ref,targets,dy),torch.autograd.grad(out,targets,dy)):
        assert (ideal.float()-got.float()).norm() <= .012*ideal.float().norm()+1e-5


@pytest.mark.parametrize('backend',['direct','packed'])
def test_zero_start_opens_and_restarts(setting,backend,tmp_path):
    c = replace(setting,chord_history_correction=True,chord_attnres_backend=backend)
    seed_all(42)
    base = make(replace(setting,chord_attnres=False),compiled=True)
    seed_all(42)
    model = make(c,compiled=True)
    ids, labels = inputs()
    assert model.depth_reads[0][0].scorer is None
    assert model.depth_reads[1][0].scorer is not None
    for name,tensor in base.state_dict().items():
        assert torch.equal(tensor,model.state_dict()[name]),name
    with torch.no_grad():
        logits = model(ids)
        assert torch.equal(logits,base(ids))
    opt,_ = optimizer_for(model)
    model(ids,labels).backward()
    for layer in model.depth_reads:
        for reader in layer:
            assert reader.gain.item()==0 and reader.gain.grad.abs().item()>0
            if reader.scorer is not None:
                assert all(torch.count_nonzero(p.grad)==0 for p in reader.scorer.parameters())
    opt.step()
    opt.zero_grad(set_to_none=True)
    assert all(r.gain.abs().item()>0 for layer in model.depth_reads for r in layer)
    model(ids,labels).backward()
    for layer in model.depth_reads:
        for reader in layer:
            if reader.scorer is not None:
                assert all(torch.isfinite(p.grad).all() and p.grad.norm()>0 for p in reader.scorer.parameters())
    # Test meaningful corrections of both signs instead of just first-step gains.
    with torch.no_grad():
        for ra,rm in model.depth_reads:
            ra.gain.fill_(.65); rm.gain.fill_(-.4)
        full = model(ids)
        cache = KVCache(2,1,32,128,c.depth,'cuda',torch.bfloat16)
        parts = torch.cat([model(ids[:,:7],kv_cache=cache), model(ids[:,7:],kv_cache=cache)], dim=1)
        assert (parts-full).norm() < .025*full.norm()
        changed=ids.clone(); changed[:,7:]=(changed[:,7:]+1)%c.vocab_size
        assert torch.equal(full[:,:7],model(changed)[:,:7])
    check_resume(c,tmp_path)


@pytest.mark.parametrize('backend',['direct','packed'])
def test_history_only_reference(setting,backend):
    model = make(replace(setting,chord_history_correction=True,chord_attnres_backend=backend))
    for j,layer in enumerate(model.depth_reads):
        for reader in layer:
            k = j+1
            endpoints = tuple(torch.randn(2,13,128,device='cuda',dtype=torch.bfloat16).requires_grad_() for _ in range(k))
            bank = torch.stack([x.flatten(0,1) for x in endpoints],dim=1)
            data = bank if backend=='packed' else endpoints
            actual = reader(data,backend=='packed',endpoints[0].shape)
            if k==1:
                assert torch.equal(actual,endpoints[0])
                continue
            scorer=reader.scorer
            ref=reference_mix(bank[:,-1],bank[:,:-1],scorer.norm_weight,scorer.proj_weight).view_as(actual)
            torch.testing.assert_close(actual,ref,rtol=.008,atol=.008)
            dy=torch.randn_like(actual)
            leaves=(*endpoints,scorer.norm_weight,scorer.proj_weight)
            for ideal,got in zip(torch.autograd.grad(ref,leaves,dy),torch.autograd.grad(actual,leaves,dy)):
                assert (ideal.float()-got.float()).norm() <= .012*ideal.float().norm()+1e-5
