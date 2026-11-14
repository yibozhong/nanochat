"""Embedding-query and independent-per-block QKV latent feedback variants."""
import torch
from torch import nn
from torch.nn import functional as F

from nanochat.connection_model import ConnectionModel
from nanochat.gpt import Linear,norm
from nanochat.latent_feedback import FeedbackKVCache,sampled_passes
from nanochat.projected_layer_read import projected_read


def normalize_bank(bank):
    # Only the raw previous-pass backbone state is detached. K/V projections
    # below run under the final pass's autograd context and remain trainable.
    return F.rms_norm(bank.detach().float(),(bank.shape[-1],),eps=1e-5).to(bank.dtype)


class ProjectedLayerReader(nn.Module):
    def __init__(self,width,backend):
        super().__init__();self.backend=backend
        self.q_proj=Linear(width,width,bias=False)
        self.k_proj=Linear(width,width,bias=False)
        self.v_proj=Linear(width,width,bias=False)
        self.o_proj=Linear(width,width,bias=False)
        for p in self.parameters():nn.init.normal_(p,std=.02)

    def forward(self,x,bank,shift=True,normalize_query=False):
        q=self.q_proj(norm(x) if normalize_query else x)
        # Independent K and V parameters, packed into one large GEMM.
        weight=torch.cat((self.k_proj.weight,self.v_proj.weight),0).to(bank.dtype)
        kv=F.linear(bank,weight)
        read=projected_read(q,kv,shift=shift,backend=self.backend)
        return x+self.o_proj(read)


class ProjectedFeedbackModel(ConnectionModel):
    def __init__(self,config):
        super().__init__(config)
        count=config.depth if config.feedback_source=='per_layer_qkv' else 1
        self.feedback_readers=nn.ModuleList([ProjectedLayerReader(config.width,config.feedback_attention_backend) for _ in range(count)])
        self.compiled_readers=None;self.compiled_bank_norm=None

    def feedback_parameters(self):
        return self.feedback_readers.parameters()

    @staticmethod
    def training_passes(seed,step):return sampled_passes(seed,step)

    def prepare(self,device='cuda',dtype=torch.bfloat16,compile_blocks=False):
        super().prepare(device,dtype,compile_blocks)
        if compile_blocks:
            self.compiled_readers=[torch.compile(r.forward,dynamic=False,fullgraph=True) for r in self.feedback_readers]
            self.compiled_bank_norm=torch.compile(normalize_bank,dynamic=False,fullgraph=True)
        return self

    def _pass(self,embedding,bank=None,collect=False,kv_cache=None):
        if bank is None:
            return self.backbone(embedding,kv_cache,return_layers=collect)
        bank=(self.compiled_bank_norm if self.compiled_bank_norm is not None and kv_cache is None else normalize_bank)(bank)
        shift=kv_cache is None
        def read(i,x):
            fn=self.compiled_readers[i] if self.compiled_readers is not None and kv_cache is None else self.feedback_readers[i]
            return fn(x,bank,shift=shift,normalize_query=i>0)
        if self.config.feedback_source=='embedding_qkv':
            return self.backbone(read(0,embedding),kv_cache,return_layers=collect)
        return self.backbone(embedding,kv_cache,return_layers=collect,before_block=read)

    def hidden(self,idx,kv_cache=None,passes=None):
        if idx.ndim!=2 or idx.shape[1]<1:raise ValueError('Expected nonempty [batch,tokens] input')
        if kv_cache is not None:
            if passes is not None:raise ValueError('Cached decoding uses exact recurrence, not parallel passes')
            if torch.is_grad_enabled():raise ValueError('Cached decoding requires no_grad/inference_mode')
            if not isinstance(kv_cache,FeedbackKVCache):raise TypeError('Use FeedbackKVCache')
            if kv_cache.get_pos()+idx.shape[1]>self.config.sequence_length:raise ValueError('Input exceeds configured context')
            if kv_cache.get_pos() and kv_cache.feedback_state is None:raise ValueError('Nonempty cache lacks feedback state')
            states=[]
            for token in idx.split(1,dim=1):
                e=norm(self.wte(token))
                bank=kv_cache.feedback_state if kv_cache.get_pos() else None
                top,new_bank=self._pass(e,bank,collect=True,kv_cache=kv_cache)
                kv_cache.feedback_state=new_bank.detach();states.append(top)
            return norm(torch.cat(states,1))
        passes=(2 if self.training else self.config.feedback_eval_passes) if passes is None else passes
        if type(passes) is not int or passes<1:raise ValueError('passes must be a positive integer')
        embedding=norm(self.wte(idx));bank=None;grad_enabled=torch.is_grad_enabled()
        for index in range(passes):
            final=index==passes-1
            with torch.set_grad_enabled(grad_enabled and final):
                if final:top=self._pass(embedding,bank)
                else:top,bank=self._pass(embedding,bank,collect=True)
        return norm(top)

    def forward(self,idx,targets=None,dense=False,kv_cache=None,passes=None):
        return self.readout(self.hidden(idx,kv_cache,passes),targets,dense)
