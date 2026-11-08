"""Deterministic, coalesced reduction of small-parameter gradient workspaces."""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _scale_partial_reduce(P, O, K: tl.constexpr, S: tl.constexpr,
                          R: tl.constexpr, BLOCK_K: tl.constexpr):
    k=tl.program_id(0)*BLOCK_K+tl.arange(0,BLOCK_K)
    block=tl.program_id(1);group=tl.program_id(2)
    rows=block*R+tl.arange(0,R)
    x=tl.load(P+group*S*K+rows[:,None]*K+k[None,:],
              (rows[:,None]<S)&(k[None,:]<K),0)
    blocks=tl.cdiv(S,R)
    tl.store(O+group*blocks*K+block*K+k,tl.sum(x,0),k<K)


def reduce_scale_partials(parts):
    # parts [number_of_scales, token_CTAs, channel, stream].
    groups,s,c,n=parts.shape;k=c*n
    out=torch.empty((groups,c,n),device=parts.device,dtype=torch.float32)
    if s<=128:
        _scale_partial_reduce[(tr.cdiv(k,64),1,groups)](
            parts,out,k,s,tr.next_power_of_2(s),64,num_warps=4)
    else:
        rows=tr.cdiv(s,128)
        temporary=torch.empty((groups,rows,k),device=parts.device,dtype=torch.float32)
        _scale_partial_reduce[(tr.cdiv(k,64),rows,groups)](
            parts,temporary,k,s,128,64,num_warps=4)
        _scale_partial_reduce[(tr.cdiv(k,64),1,groups)](
            temporary,out,k,rows,tr.next_power_of_2(rows),64,num_warps=4)
    return out
