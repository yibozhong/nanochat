# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# Adapted from FLA parallel attention (MIT); see
# research/connections/dilated-attention/source.json and LICENSE.fla.
# New sparse traversal: real prefix tokens, dense local window, strided tail.
import triton as tr
import triton.language as tl


@tr.jit
def _query_tile(q, go, K, V, rows, cols, valid, base, H:tl.constexpr,D:tl.constexpr,
                T:tl.constexpr,scale:tl.constexpr, lse,delta,m,l,acc,BWD:tl.constexpr):
    dims=tl.arange(0,D)
    key=tl.load(K+base+cols[:,None]*H*D+dims[None,:],
                (cols[:,None]>=0)&(cols[:,None]<T),0)
    value=tl.load(V+base+cols[:,None]*H*D+dims[None,:],
                  (cols[:,None]>=0)&(cols[:,None]<T),0)
    scores=tl.dot(q,tl.trans(key))*scale*1.4426950408889634
    valid=valid&(rows[:,None]<T)&(cols[None,:]>=0)&(cols[None,:]<T)
    scores=tl.where(valid,scores,float('-inf'))
    if BWD:
        p=tl.exp2(scores-lse[:,None])
        dp=tl.dot(go,tl.trans(value))
        ds=p*(dp-delta[:,None])
        acc=tl.dot(ds.to(q.dtype),key,acc)
    else:
        new_m=tl.maximum(m,tl.max(scores,axis=1))
        pivot=tl.where(new_m==float('-inf'),0.,new_m)
        alpha=tl.exp2(m-pivot)
        p=tl.exp2(scores-pivot[:,None])
        l=l*alpha+tl.sum(p,axis=1)
        acc=acc*alpha[:,None]+tl.dot(p.to(q.dtype),value)
        m=new_m
    return m,l,acc


@tr.jit
def sparse_query(Q,K,V,O,LSE,GO,DELTA,DQ,T:tl.constexpr,H:tl.constexpr,D:tl.constexpr,
                 C:tl.constexpr,R:tl.constexpr,S:tl.constexpr,STRIDE:tl.constexpr,
                 BM:tl.constexpr,BN:tl.constexpr,BWD:tl.constexpr):
    bh=tl.program_id(1);residue=tl.program_id(2)
    first=tl.program_id(0)*BM*STRIDE+residue
    rows=first+tl.arange(0,BM)*STRIDE
    dims=tl.arange(0,D);base=(bh//H*T*H+bh%H)*D
    q=tl.load(Q+base+rows[:,None]*H*D+dims[None,:],rows[:,None]<T,0)
    if BWD:
        go=tl.load(GO+base+rows[:,None]*H*D+dims[None,:],rows[:,None]<T,0)
        lse=tl.load(LSE+bh//H*T*H+rows*H+bh%H,rows<T,0)
        delta=tl.load(DELTA+bh//H*T*H+rows*H+bh%H,rows<T,0)
    else:
        go=q;lse=tl.full((BM,),0.,tl.float32);delta=lse
    m=tl.full((BM,),float('-inf'),tl.float32)
    l=tl.full((BM,),0.,tl.float32);acc=tl.full((BM,D),0.,tl.float32)
    scale:tl.constexpr=D**-.5
    # All three phases share the same softmax state. Each key appears once.
    sink=tl.arange(0,16)
    sink_valid=(sink[None,:]<S)&(sink[None,:]<=rows[:,None])
    m,l,acc=_query_tile(q,go,K,V,rows,sink,sink_valid,base,H,D,T,scale,lse,delta,m,l,acc,BWD)
    last=tl.minimum(T-1,first+(BM-1)*STRIDE)
    for start in range(tl.maximum(S,first-C+1),last+1,BN):
        cols=start+tl.arange(0,BN)
        distance=rows[:,None]-cols[None,:]
        valid=(cols[None,:]>=S)&(distance>=0)&(distance<C)
        m,l,acc=_query_tile(q,go,K,V,rows,cols,valid,base,H,D,T,scale,lse,delta,m,l,acc,BWD)
    origin=first-C-(R-1)*STRIDE
    begin=tl.maximum(0,(S-origin+STRIDE-1)//STRIDE)
    for start in range(begin,R+BM-1,BN):
        cols=origin+(start+tl.arange(0,BN))*STRIDE
        distance=rows[:,None]-cols[None,:]
        valid=(cols[None,:]>=S)&(distance>=C)&(distance<=C+(R-1)*STRIDE)
        m,l,acc=_query_tile(q,go,K,V,rows,cols,valid,base,H,D,T,scale,lse,delta,m,l,acc,BWD)
    offsets=base+rows[:,None]*H*D+dims[None,:]
    if BWD:
        tl.store(DQ+offsets,acc*scale,rows[:,None]<T)
    else:
        tl.store(O+offsets,acc/tl.where(l>0,l,1.)[:,None],rows[:,None]<T)
        tl.store(LSE+bh//H*T*H+rows*H+bh%H,m+tl.log2(l),rows<T)


@tr.jit
def sparse_delta(O,GO,DELTA,N:tl.constexpr,D:tl.constexpr,BM:tl.constexpr):
    rows=tl.program_id(0)*BM+tl.arange(0,BM);dims=tl.arange(0,D)
    o=tl.load(O+rows[:,None]*D+dims[None,:],rows[:,None]<N,0).to(tl.float32)
    go=tl.load(GO+rows[:,None]*D+dims[None,:],rows[:,None]<N,0).to(tl.float32)
    tl.store(DELTA+rows,tl.sum(o*go,axis=1),rows<N)


@tr.jit
def _kv_tile(key,value,Q,GO,LSE,DELTA,rows,cols,valid,base,bh,
             T:tl.constexpr,H:tl.constexpr,D:tl.constexpr,dk,dv):
    dims=tl.arange(0,D)
    q=tl.load(Q+base+cols[:,None]*H*D+dims[None,:],(cols[:,None]>=0)&(cols[:,None]<T),0)
    go=tl.load(GO+base+cols[:,None]*H*D+dims[None,:],(cols[:,None]>=0)&(cols[:,None]<T),0)
    lse=tl.load(LSE+bh//H*T*H+cols*H+bh%H,(cols>=0)&(cols<T),0)
    delta=tl.load(DELTA+bh//H*T*H+cols*H+bh%H,(cols>=0)&(cols<T),0)
    scores=tl.dot(key,tl.trans(q))*(D**-.5)*1.4426950408889634
    valid=valid&(rows[:,None]<T)&(cols[None,:]>=0)&(cols[None,:]<T)
    p=tl.exp2(tl.where(valid,scores-lse[None,:],float('-inf')))
    dp=tl.dot(value,tl.trans(go));ds=p*(dp-delta[None,:])
    dk=tl.dot(ds.to(key.dtype),q,dk)
    dv=tl.dot(p.to(key.dtype),go,dv)
    return dk,dv


@tr.jit
def sparse_dkv(Q,K,V,GO,LSE,DELTA,DK,DV,T:tl.constexpr,H:tl.constexpr,D:tl.constexpr,
               C:tl.constexpr,R:tl.constexpr,S:tl.constexpr,STRIDE:tl.constexpr,
               BM:tl.constexpr,BN:tl.constexpr,SINK:tl.constexpr):
    bh=tl.program_id(1);base=(bh//H*T*H+bh%H)*D
    if SINK:
        first=0;rows=tl.arange(0,BM);active=(rows<S)&(rows<T)
    else:
        first=tl.program_id(0)*BM*STRIDE+tl.program_id(2)
        rows=first+tl.arange(0,BM)*STRIDE;active=(rows>=S)&(rows<T)
    dims=tl.arange(0,D)
    offsets=base+rows[:,None]*H*D+dims[None,:]
    key=tl.load(K+offsets,active[:,None],0);value=tl.load(V+offsets,active[:,None],0)
    dk=tl.full((BM,D),0.,tl.float32);dv=tl.full((BM,D),0.,tl.float32)
    if SINK:
        for start in range(0,T,BN):
            cols=start+tl.arange(0,BN)
            valid=active[:,None]&(cols[None,:]>=rows[:,None])
            dk,dv=_kv_tile(key,value,Q,GO,LSE,DELTA,rows,cols,valid,base,bh,T,H,D,dk,dv)
    else:
        last=tl.minimum(T-1,first+(BM-1)*STRIDE)
        for start in range(first,tl.minimum(T,last+C),BN):
            cols=start+tl.arange(0,BN)
            distance=cols[None,:]-rows[:,None]
            valid=active[:,None]&(distance>=0)&(distance<C)
            dk,dv=_kv_tile(key,value,Q,GO,LSE,DELTA,rows,cols,valid,base,bh,T,H,D,dk,dv)
        origin=first+C
        end=tl.minimum(R+BM-1,tl.maximum(0,(T-origin+STRIDE-1)//STRIDE))
        for start in range(0,end,BN):
            cols=origin+(start+tl.arange(0,BN))*STRIDE
            distance=cols[None,:]-rows[:,None]
            valid=active[:,None]&(distance>=C)&(distance<=C+(R-1)*STRIDE)
            dk,dv=_kv_tile(key,value,Q,GO,LSE,DELTA,rows,cols,valid,base,bh,T,H,D,dk,dv)
    tl.store(DK+offsets,dk*(D**-.5),active[:,None])
    tl.store(DV+offsets,dv,active[:,None])
