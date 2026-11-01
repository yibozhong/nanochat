"""Full-attention Transformer branches; all depth propagation uses updated FW."""
from dataclasses import replace
from functools import partial
import math

import torch
from torch.utils.checkpoint import checkpoint

from nanochat.connection_model import ConnectionModel
from nanochat.depth_fast_weights import DepthFastWeights
from nanochat.gpt import norm


class DFWRModel(ConnectionModel):
    def __init__(self, config):
        # Reuse vanilla modules/initialization and the existing full-vocab head.
        super().__init__(replace(config,variant='baseline'))
        self.config = config
        with torch.random.fork_rng(devices=[]):
            self.depth_memory = DepthFastWeights(config.width,config.fw_bottleneck,
                config.fw_inner_lr,config.fw_momentum,config.fw_ns_steps,
                config.fw_ns_eps,config.fw_ns_bf16,config.fw_learnable_lr)
            # Zero branch outputs make the very first normalized inner update
            # singularly sensitive. Small nonzero outputs avoid that start.
            for block in self.blocks:
                torch.nn.init.normal_(block.attn.c_proj.weight,std=.1/math.sqrt(config.width))
                torch.nn.init.normal_(block.mlp.c_proj.weight,std=.1/math.sqrt(4*config.width))
        self.compiled_update = None

    def prepare(self,device='cuda',dtype=torch.bfloat16,compile_blocks=False):
        self.to(device)
        self.wte.to(dtype=dtype)
        self.cos=self.cos.to(dtype);self.sin=self.sin.to(dtype)
        if compile_blocks:
            for block in self.blocks:
                block.compiled_attention=torch.compile(block.attention_branch,dynamic=False,fullgraph=True)
                block.compiled_mlp=torch.compile(block.mlp_branch,dynamic=False,fullgraph=True)
            self.compiled_update=torch.compile(self.depth_memory.update_read,dynamic=False,fullgraph=True)
        return self

    def _block_step(self,block,h,theta,momentum,cos,sin,kv_cache=None):
        update = self.compiled_update or self.depth_memory.update_read
        delta = (block.compiled_attention or block.attention_branch)(h,cos,sin,kv_cache)
        h,theta,momentum = update(h,delta,theta,momentum)
        delta = (block.compiled_mlp or block.mlp_branch)(h)
        return update(h,delta,theta,momentum)

    def hidden(self,idx,kv_cache=None,*,return_streams=False):
        if return_streams:
            raise ValueError('DFWR has token-local FW states, not mHC streams')
        start=0 if kv_cache is None else kv_cache.get_pos()
        if start+idx.shape[1]>self.config.sequence_length:
            raise ValueError('Input exceeds configured context length')
        h=norm(self.wte(idx))
        h,theta,momentum=self.depth_memory.initialize(h)
        cos,sin=self.cos[:,start:start+idx.shape[1]],self.sin[:,start:start+idx.shape[1]]
        for block in self.blocks:
            fn=partial(self._block_step,block)
            if self.config.activation_checkpoint and self.training and torch.is_grad_enabled():
                if kv_cache is not None:
                    raise ValueError('Checkpoint recomputation cannot mutate a KV cache')
                h,theta,momentum=checkpoint(fn,h,theta,momentum,cos,sin,use_reentrant=False)
            else:
                h,theta,momentum=fn(h,theta,momentum,cos,sin,kv_cache)
        # State is deliberately local to this call, including incremental decode.
        return norm(h)
