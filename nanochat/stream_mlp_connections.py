"""Original mHC read/write with a channel-shared 4->16->4 ReLU mixer."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from nanochat.nvidia_connections import interleave_phi, check_precision
from nanochat.nvidia_ops import nvidia_ops
from nanochat.stream_mlp import stream_mlp


class StreamMLPHyperConnection(nn.Module):
    def __init__(self, width, streams, branch_index, iterations=20, *,
                 use_tf32=True, fused_grad_acc=False, compact_routing=False, stream_mlp_kernel='legacy', **unused):
        super().__init__()
        if streams != 4 or fused_grad_acc:
            raise ValueError('Stream MLP requires four streams and unfused gradient accumulation')
        self.width, self.streams = width, streams
        self.use_tf32, self.sinkhorn = use_tf32, False
        self.compact_routing = compact_routing
        self.stream_mlp_kernel = stream_mlp_kernel
        self.phi = nn.Parameter(torch.empty(8, 4*width))
        self.bias = nn.Parameter(torch.empty(8))
        self.scales = nn.Parameter(torch.empty(2))
        self.mix_in = nn.Parameter(torch.empty(16, 4))
        self.mix_out = nn.Parameter(torch.empty(4, 16))

    @torch.no_grad()
    def reset_parameters(self):
        # Preserve original routing and all subsequent initialization RNG draws.
        original = self.phi.new_empty(24, 4*self.width)
        nn.init.normal_(original, std=1/math.sqrt(4*self.width))
        self.phi.copy_(interleave_phi(original)[:8])
        self.bias[:4].fill_(-math.log(3)); self.bias[4:].zero_(); self.scales.fill_(.01)
        eye = torch.eye(4, device=self.phi.device)
        # Two distinct orthogonal bases. Every row has a nonzero sum, so even
        # the initially replicated streams exercise all 16 units across signs.
        q = self.phi.new_tensor([[1,1,1,-1], [1,-1,1,1], [1,1,-1,1], [1,-1,-1,-1]])*.5
        self.mix_in.copy_(torch.cat((eye, -eye, q, -q), dim=0))
        self.mix_out.copy_(.5*torch.cat((eye, -eye, q.T, -q.T), dim=1))
        # .5 lives only in learned initial weights; no runtime scaling/bypass.

    def read(self, x):
        check_precision(self.use_tf32)
        x = x.contiguous(); b,t,c,n = x.shape; ops = nvidia_ops()
        if self.compact_routing:
            from nanochat.compact_routing import compact_routing
            h=compact_routing(x.reshape(b*t,c*n),self.phi,self.scales,self.bias,use_tf32=self.use_tf32)
            pre=h[:,:4].contiguous();post=h[:,4:].contiguous()
            branch=ops.mhc_fused_aggregate(x,pre.reshape(b,t,n),n,
                use_tf32=self.use_tf32,fused_grad_x_acc_buffer=None)
            return branch,x,post.reshape(b,t,n)
        raw, ms = ops.mhc_fused_projection(x.reshape(b*t,c*n), F.pad(self.phi,(0,0,0,16)),
            norm_weight=None, use_tf32=self.use_tf32, fused_grad_x_acc_buffer=None, use_split_k=False)
        pre, post, _ = ops.mhc_fused_scale(raw, F.pad(self.scales,(0,1)),
            F.pad(self.bias,(0,16)).unsqueeze(0), ms, n)
        branch = ops.mhc_fused_aggregate(x, pre.reshape(b,t,n), n,
            use_tf32=self.use_tf32, fused_grad_x_acc_buffer=None)
        return branch, x, post.reshape(b,t,n).contiguous()

    def write(self, output, residual, post):
        if self.stream_mlp_kernel != 'legacy':
            from nanochat.stream_mlp_optimized import optimized_stream_mlp
            return optimized_stream_mlp(residual, output.contiguous(), post, self.mix_in, self.mix_out, self.stream_mlp_kernel)
        return stream_mlp(residual, output.contiguous(), post, self.mix_in, self.mix_out)
