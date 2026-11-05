"""Analytic matrix FLOPs for the retained full-attention vanilla model."""


def model_cost(config, batch=1, activation_bytes=2):
    c=config
    if c.architecture!='vanilla' or c.variant not in ('baseline', 'mhc_stream_mlp'):
        return None
    t,d,depth=c.sequence_length,c.width,c.depth
    projections=24*batch*t*depth*d*d
    head=2*batch*t*d*c.vocab_size
    attention=4*batch*d*depth*(t*(t+1)//2)
    components=dict(backbone_projections=projections,vocab_head=head,attention=attention)
    if c.variant=='mhc_stream_mlp':
        tokens=batch*t; connectors=2*depth
        components['stream_mlp']=connectors*tokens*d*2*(4*16+16*4)
        components['routing_projection_logical']=connectors*tokens*2*(4*d)*8+tokens*2*(4*d)*4
        components['aggregation_distribution']=connectors*tokens*4*(4*d)+tokens*2*(4*d)
    forward=sum(components.values())
    return dict(estimated_forward_flops=forward,estimated_training_flops=3*forward,
        components_forward=components,
        accounting='FMA=2 FLOPs; allowed attention pairs, not masked dense-kernel execution. Backward approximated as2x consumed forward matrix FLOPs. Excludes softmax/norm/activation/optimizer/recomputation and is not measured hardware FLOPs.',
        memory_note='Actual training peak VRAM must be measured separately.')
