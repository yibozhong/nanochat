"""Experimental fused forward and VJP of nanochat.geodesic_residual.geodesic_split_write."""
import math

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _mean(x, D: tl.constexpr):
    return tl.sum(x, 0) / D


@triton.jit
def _quarter_turn(x, D: tl.constexpr):
    i = tl.arange(0, D)
    return tl.gather(x, i ^ (D // 2), 0) * tl.where(i < D // 2, -1., 1.)


@triton.jit
def _log(h, f, FLOOR2: tl.constexpr, D: tl.constexpr):
    r = _mean(h*f, D)
    t = f - r*h
    z = _mean(t*t, D)
    # Near an antipode the log amplifies radial roundoff by ~pi/sqrt(z).
    # Retain the dot and its cancellation in double only on affected rows.
    if (r < 0.) & (z < 1e-4*r*r):
        hd = h.to(tl.float64)
        fd = f.to(tl.float64)
        radial = _mean(hd*fd, D)
        t = (fd-radial*hd).to(tl.float32)
        z = _mean(t*t, D)
        r = radial.to(tl.float32)
    f2 = z + r*r
    weight = tl.minimum(f2/FLOOR2, 1.)
    threshold = 1e-12*tl.maximum(f2, FLOOR2)
    transverse = tl.sqrt(tl.maximum(z, threshold))
    angle = libdevice.atan2(transverse, r)
    k = weight*angle/transverse
    cut = (r < 0.) & (z <= threshold)
    if cut:
        log = (3.141592653589793*weight)*_quarter_turn(h, D)
    else:
        log = k*t
    return log, r, t, z, f2, weight, threshold, transverse, angle, k, cut


@triton.jit
def _split(log, v, a, b, BETA: tl.constexpr, D: tl.constexpr):
    v2 = _mean(v*v, D)
    den = tl.maximum(v2, 1e-12)
    proj = _mean(log*v, D)/den
    s = a*log + ((b-a)*proj)*v
    m = BETA*v + (1.-BETA)*s
    return s, m, v2, den, proj


@triton.jit
def _exp(h, s, m, D: tl.constexpr):
    u = _mean(s*s, D)
    angle = tl.sqrt(tl.maximum(u, 1e-8))
    half = .5*angle
    sinc_half = libdevice.sin(half)/half
    sinc = tl.where(u < 1e-8, 1.-u/6., libdevice.sin(angle)/angle)
    cosc = tl.where(u < 1e-8, -.5+u/24., -.5*(sinc_half*sinc_half))
    c = 1.+u*cosc
    trial = c*h+sinc*s
    norm2 = _mean(trial*trial, D)
    inv = tl.rsqrt(tl.maximum(norm2, 1e-12))
    nh = trial*inv
    dot = _mean(m*s, D)
    shift = cosc*s-sinc*h
    carried = m+dot*shift
    radial = _mean(nh*carried, D)
    return nh, carried, radial, dot, shift, inv, norm2, u, angle, sinc_half, sinc, cosc, c


@triton.jit
def _forward(H,V,F,A,B,NH,NV, BETA:tl.constexpr, FLOOR2:tl.constexpr, D:tl.constexpr):
    ix = tl.program_id(0).to(tl.int64)*D + tl.arange(0,D)
    h, v, f = tl.load(H+ix), tl.load(V+ix), tl.load(F+ix).to(tl.float32)
    log = _log(h,f,FLOOR2,D)[0]
    s,m,_,_,_ = _split(log,v,tl.load(A),tl.load(B),BETA,D)
    e = _exp(h,s,m,D)
    nh, carried, radial = e[0],e[1],e[2]
    tl.store(NH+ix, nh)
    tl.store(NV+ix, carried-radial*nh)


@triton.jit
def _backward(H,V,F,A,B,GH,GV,DH,DV,DF,DA,DB, BETA:tl.constexpr, FLOOR2:tl.constexpr, D:tl.constexpr):
    row = tl.program_id(0)
    ix = row.to(tl.int64)*D + tl.arange(0,D)
    h,v,f = tl.load(H+ix),tl.load(V+ix),tl.load(F+ix).to(tl.float32)
    a,b = tl.load(A),tl.load(B)
    log,r,t,z,f2,weight,threshold,transverse,log_angle,k,cut = _log(h,f,FLOOR2,D)
    s,m,v2,den,proj = _split(log,v,a,b,BETA,D)
    nh,carried,radial,dot,shift,inv,norm2,u,angle,sinc_half,sinc,cosc,c = _exp(h,s,m,D)

    # Scalar adjoints below use a channel mean; the two shared parameters
    # receive channel sums and subsequently a deterministic sum over tokens.
    gv = tl.load(GV+ix)
    q = _mean(gv*nh,D)
    dc = gv-q*nh
    dnh = tl.load(GH+ix)-radial*gv-q*carried
    dy = inv*(dnh-tl.where(norm2 >= 1e-12,_mean(dnh*nh,D)*nh,0.))
    ddot = _mean(dc*shift,D)
    dm = dc+ddot*s
    ds = ddot*m+(dot*cosc)*dc+sinc*dy
    dh = c*dy-(dot*sinc)*dc
    dc_scalar = _mean(dy*h,D)
    dcosc = dot*_mean(dc*s,D)+u*dc_scalar
    dsinc = _mean(dy*s,D)-dot*_mean(dc*h,D)
    u_safe = tl.maximum(u,1e-8)
    sinc_prime = tl.where(u<1e-8,-1./6.,(libdevice.cos(angle)-sinc)/ (2.*u_safe))
    cosc_prime = tl.where(u<1e-8,1./24.,-sinc_half*(libdevice.cos(.5*angle)-sinc_half)/(2.*u_safe))
    du = dc_scalar*cosc+dsinc*sinc_prime+dcosc*cosc_prime
    ds += (2.*du)*s+(1.-BETA)*dm
    dv = BETA*dm

    dcoeff = _mean(ds*v,D)
    dproj = (b-a)*dcoeff/den
    dl = a*ds + dproj*v
    dv += ((b-a)*proj)*ds + dproj*(log-tl.where(v2>=1e-12,2.*proj*v,0.))
    tl.store(DA+row,tl.sum(ds*(log-proj*v),0))
    tl.store(DB+row,tl.sum(ds*(proj*v),0))

    if cut:
        dweight = 3.141592653589793*_mean(dl*_quarter_turn(h,D),D)
        dh -= (3.141592653589793*weight)*_quarter_turn(dl,D)
        dt = tl.full((D,),0,tl.float32)
        dz = 0.
        dr = 0.
        df2_from_floor = 0.
    else:
        dt = k*dl
        dk = _mean(dl*t,D)
        dweight = dk*log_angle/transverse
        dangle = dk*weight/transverse
        inv_hypot2 = 1./(transverse*transverse+r*r)
        dr = -dangle*transverse*inv_hypot2
        dtransverse = dangle*r*inv_hypot2-dk*k/transverse
        dmax = .5*dtransverse/transverse
        # torch.maximum splits ties; clamp_min and clamp(max=...) include them.
        to_z = tl.where(z>threshold,1.,tl.where(z==threshold,.5,0.))
        dz = dmax*to_z
        df2_from_floor = tl.where(f2>=FLOOR2,dmax*(1.-to_z)*1e-12,0.)
    df2 = df2_from_floor + tl.where(f2<=FLOOR2,dweight/FLOOR2,0.)
    dt += (2.*(dz+df2))*t
    dr += 2.*r*df2-_mean(dt*h,D)
    tl.store(DF+ix,dt+dr*h)
    tl.store(DH+ix,dh+dr*f-r*dt)
    tl.store(DV+ix,dv)


class _GeodesicWrite(torch.autograd.Function):
    @staticmethod
    def forward(ctx,h,v,f,a,b,decay,floor):
        width = h.size(-1)
        assert width in (128,512) and h.is_cuda and h.dtype == v.dtype == a.dtype == b.dtype == torch.float32
        assert h.shape == v.shape == f.shape and f.dtype in (torch.bfloat16,torch.float32)
        assert a.numel() == b.numel() == 1 and math.isfinite(floor) and floor>0
        nh,nv = torch.empty_like(h),torch.empty_like(v)
        _forward[(h.numel()//width,)](h,v,f,a,b,nh,nv,decay,floor**2,width,num_warps=4,enable_fp_fusion=False)
        ctx.save_for_backward(h,v,f,a,b)
        ctx.decay,ctx.floor2 = decay,floor**2
        return nh,nv

    @staticmethod
    def backward(ctx,gh,gv):
        h,v,f,a,b = ctx.saved_tensors
        width = h.size(-1)
        rows = h.numel()//width
        dh,dv,df = torch.empty_like(h),torch.empty_like(v),torch.empty_like(f)
        da,db = h.new_empty((rows,)),h.new_empty((rows,))
        _backward[(rows,)](h,v,f,a,b,gh.contiguous(),gv.contiguous(),dh,dv,df,da,db,
                            ctx.decay,ctx.floor2,width,num_warps=4,enable_fp_fusion=False)
        return dh,dv,df,da.sum().reshape_as(a),db.sum().reshape_as(b),None,None


def fused_geodesic_write(h,v,force,perpendicular,parallel,decay,rms_floor):
    return _GeodesicWrite.apply(h.contiguous(),v.contiguous(),force.contiguous(),
                                perpendicular,parallel,decay,rms_floor)
