"""Content-conditioned attention over the previous token's detached layer bank.

query: [B,T,D]; bank: [B,T,L,D], raw states after complete blocks.
Keys and values are the same FP32 RMS-normalized states (epsilon 1e-5).
Only query gradients are required: the preliminary pass is explicitly detached.
One Triton program per token fuses normalization, scores, depth softmax and read.
Backward reuses FP32 probabilities, recomputes normalization and writes only dQ.
"""
import torch
import triton
import triton.language as tl


def layer_read_reference(query, bank, shift=True, normalize=True):
    dtype=torch.float64 if query.dtype==torch.float64 else torch.float32
    value=bank.to(dtype)
    if normalize:
        value=value*torch.rsqrt(value.square().mean(-1,keepdim=True)+1e-5)
    if shift:
        value=torch.cat((torch.zeros_like(value[:,:1]),value[:,:-1]),dim=1)
    scores=(query.to(dtype).unsqueeze(2)*value).sum(-1)*(query.shape[-1]**-.5)
    weights=scores.softmax(-1)
    return (weights.unsqueeze(-1)*value).sum(2).to(query.dtype)


@triton.jit
def _read_forward(Q, BANK, OUT, PROBS, T:tl.constexpr, L:tl.constexpr, D:tl.constexpr,
                  SHIFT:tl.constexpr, NORMALIZE:tl.constexpr, BL:tl.constexpr, BD:tl.constexpr):
    row=tl.program_id(0)
    valid=(row%T>0) if SHIFT else True
    previous=row-1 if SHIFT else row
    layers=tl.arange(0,BL)
    channels=tl.arange(0,BD)
    raw=tl.load(BANK+(previous*L+layers[:,None])*D+channels[None,:],
        mask=valid & (layers[:,None]<L) & (channels[None,:]<D),other=0).to(tl.float32)
    value=raw
    if NORMALIZE:
        inv=tl.rsqrt(tl.sum(raw*raw,axis=1)/D+1e-5)
        value=raw*inv[:,None]
    query=tl.load(Q+row*D+channels,mask=channels<D,other=0).to(tl.float32)
    score=tl.sum(value*query[None,:],axis=1)*(D**-.5)
    score=tl.where(layers<L,score,float('-inf'))
    exp=tl.exp(score-tl.max(score,axis=0))
    prob=exp/tl.sum(exp,axis=0)
    prob=tl.where(valid,prob,0.)
    output=tl.sum(prob[:,None]*value,axis=0)
    tl.store(OUT+row*D+channels,output,mask=channels<D)
    tl.store(PROBS+row*L+layers,prob,mask=layers<L)


@triton.jit
def _read_backward(GO, BANK, PROBS, GQ, T:tl.constexpr, L:tl.constexpr, D:tl.constexpr,
                   SHIFT:tl.constexpr, NORMALIZE:tl.constexpr, BL:tl.constexpr, BD:tl.constexpr):
    row=tl.program_id(0)
    valid=(row%T>0) if SHIFT else True
    previous=row-1 if SHIFT else row
    layers=tl.arange(0,BL)
    channels=tl.arange(0,BD)
    raw=tl.load(BANK+(previous*L+layers[:,None])*D+channels[None,:],
        mask=valid & (layers[:,None]<L) & (channels[None,:]<D),other=0).to(tl.float32)
    value=raw
    if NORMALIZE:
        value=raw*tl.rsqrt(tl.sum(raw*raw,axis=1)/D+1e-5)[:,None]
    prob=tl.load(PROBS+row*L+layers,mask=layers<L,other=0)
    grad=tl.load(GO+row*D+channels,mask=channels<D,other=0).to(tl.float32)
    dp=tl.sum(value*grad[None,:],axis=1)
    ds=prob*(dp-tl.sum(prob*dp,axis=0))
    dq=tl.sum(value*ds[:,None],axis=0)*(D**-.5)
    tl.store(GQ+row*D+channels,dq,mask=channels<D)


class _LayerRead(torch.autograd.Function):
    @staticmethod
    def forward(ctx,query,bank,shift,normalize):
        batch,length,width=query.shape
        depth=bank.shape[2]
        out=torch.empty_like(query)
        probs=torch.empty((batch,length,depth),device=query.device,dtype=torch.float32)
        tile_l=triton.next_power_of_2(depth);tile_d=triton.next_power_of_2(width)
        warps=4 if tile_l*tile_d<=4096 else 8
        _read_forward[(batch*length,)](query,bank,out,probs,length,depth,width,shift,normalize,tile_l,tile_d,num_warps=warps)
        ctx.save_for_backward(bank,probs)
        ctx.shape=(batch,length,width,depth,shift,normalize,tile_l,tile_d,warps)
        return out

    @staticmethod
    def backward(ctx,grad):
        bank,probs=ctx.saved_tensors
        batch,length,width,depth,shift,normalize,tile_l,tile_d,warps=ctx.shape
        dq=torch.empty((batch,length,width),device=grad.device,dtype=grad.dtype)
        _read_backward[(batch*length,)](grad.contiguous(),bank,probs,dq,length,depth,width,
            shift,normalize,tile_l,tile_d,num_warps=warps)
        return dq,None,None,None


def layer_read(query, bank, shift=True, backend='triton', normalize=True):
    if query.ndim!=3 or bank.ndim!=4 or bank.shape[:2]!=query.shape[:2] or bank.shape[-1]!=query.shape[-1]:
        raise ValueError('Expected query[B,T,D] and bank[B,T,L,D]')
    if bank.requires_grad:
        raise ValueError('Layer bank must be detached; this operator computes only query gradients')
    if backend=='torch':
        return layer_read_reference(query,bank,shift,normalize)
    if backend!='triton' or not query.is_cuda or bank.device!=query.device:
        raise ValueError('Triton layer feedback requires CUDA query and bank on the same device')
    if query.dtype not in (torch.float32,torch.bfloat16,torch.float16) or bank.dtype!=query.dtype:
        raise ValueError('Query and bank must have matching FP32/BF16/FP16 dtype')
    if min(query.shape)<1 or bank.shape[2]<1:
        raise ValueError('Layer feedback dimensions must be positive')
    if triton.next_power_of_2(bank.shape[2])*triton.next_power_of_2(query.shape[-1])>32768:
        raise ValueError('Layer tile exceeds supported fused reduction size')
    return _LayerRead.apply(query.contiguous(),bank.contiguous(),shift,normalize)
