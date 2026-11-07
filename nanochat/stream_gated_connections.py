"""Independent-stream sigmoid gates before shared no-Sinkhorn HC mixing."""
import torch
from torch import nn
from nanochat.nvidia_connections import NvidiaHyperConnection
from nanochat.stream_gate_mixing import stream_gate_mix


class StreamGatedHyperConnection(NvidiaHyperConnection):
    def __init__(self,width,streams,branch_index,iterations=20,*,use_tf32=True,
                 fused_grad_acc=False,**unused):
        if fused_grad_acc:raise ValueError('Stream sigmoid gates use ordinary gradient accumulation')
        super().__init__(width,streams,branch_index,iterations,use_tf32=use_tf32,
                         fused_grad_acc=False,sinkhorn=False)
        self.gate_weight=nn.Parameter(torch.empty(streams,width,width))
        self.gate_init_seed=torch.initial_seed()+104729*(branch_index+1)

    @torch.no_grad()
    def reset_parameters(self):
        super().reset_parameters()
        # Ordinary small normal projection, no bias or identity correction.
        # Preserve the retained HC/backbone initialization RNG sequence.
        generator=torch.Generator(device=self.gate_weight.device).manual_seed(self.gate_init_seed)
        nn.init.normal_(self.gate_weight,std=.02,generator=generator)

    def write(self,output,residual,mixes):
        post,h,buffer=mixes
        assert buffer is None
        return stream_gate_mix(residual,output.contiguous(),post.contiguous(),h.contiguous(),self.gate_weight)
