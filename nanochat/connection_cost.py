"""Analytic matrix FLOPs for the retained full-attention vanilla model."""


def model_cost(config, batch=1, activation_bytes=2):
    c=config
    if c.architecture!='vanilla' or c.variant not in ('baseline', 'state_tokens'):
        return None
    t,d,depth=c.sequence_length,c.width,c.depth
    projections=24*batch*t*depth*d*d
    head=2*batch*t*d*c.vocab_size
    attention=4*batch*d*depth*(t*(t+1)//2)
    components=dict(backbone_projections=projections,vocab_head=head,attention=attention)
    if c.variant == 'state_tokens':
        # Per sublayer: read (1Q+4K+4V+1O), write (4Q+5K+5V+4O).
        components['state_projections']=2*batch*t*(2*depth)*28*d*d
        # QK and AV: read1x4, write4x5. FMA=2, heads sum to width d.
        components['state_attention']=4*batch*t*(2*depth)*(4+4*5)*d
    forward=sum(components.values())
    return dict(estimated_forward_flops=forward,estimated_training_flops=3*forward,
        components_forward=components,
        accounting='FMA=2 FLOPs; allowed attention pairs, not masked dense-kernel execution. Backward approximated as2x consumed forward matrix FLOPs. Excludes softmax/norm/activation/optimizer/recomputation and is not measured hardware FLOPs.',
        memory_note='Actual training peak VRAM must be measured separately.')
