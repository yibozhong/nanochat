"""Analytic matrix FLOPs for the retained full-attention vanilla model."""


def model_cost(config, batch=1, activation_bytes=2):
    c=config
    static=c.variant in ('mhc_static_linear','mhc_static_mlp')
    if c.architecture!='vanilla' or (c.variant!='baseline' and not static):
        return None
    t,d,depth=c.sequence_length,c.width,c.depth
    projections=24*batch*t*depth*d*d
    head=2*batch*t*d*c.vocab_size
    attention=4*batch*d*depth*(t*(t+1)//2)
    components=dict(backbone_projections=projections,vocab_head=head,attention=attention)
    if static:
        wide=4*d;tokens=batch*t
        components['static_mixing']=2*depth*tokens*(2*wide*wide if c.variant=='mhc_static_linear' else 4*wide*d)
        components['routing_projections']=2*depth*tokens*2*wide*8+tokens*2*wide*4
        components['aggregation_distribution']=2*depth*tokens*4*wide+tokens*2*wide
    forward=sum(components.values())
    return dict(estimated_forward_flops=forward,estimated_training_flops=3*forward,
        components_forward=components,
        accounting='FMA=2 FLOPs; allowed attention pairs, not masked dense-kernel execution. Backward approximated as2x consumed forward matrix FLOPs. Excludes softmax/norm/activation/optimizer/recomputation and is not measured hardware FLOPs.',
        memory_note='Actual training peak VRAM must be measured separately.')
