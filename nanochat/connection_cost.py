"""Analytic matrix FLOPs for the retained full-attention vanilla model."""


def model_cost(config, batch=1, activation_bytes=2):
    c=config
    if c.architecture!='vanilla' or c.variant!='baseline':
        return None
    t,d,depth=c.sequence_length,c.width,c.depth
    projections=24*batch*t*depth*d*d
    head=2*batch*t*d*c.vocab_size
    attention=4*batch*d*depth*(t*(t+1)//2)
    forward=projections+head+attention
    extra = {}
    if c.attention_statistic != 'none':
        # Current reference executes dense masked matrices, recomputes QK and
        # first moment (variance/Jensen), and checkpoints the statistics branch.
        stat_dense = 3*2*batch*depth*t*t*d
        if c.attention_kernel == 'fla_triton':
            stat_dense = 2*batch*depth*d*(t*(t+1)//2)
        stat_proj = 2*batch*t*depth*d*d
        forward += stat_dense + stat_proj
        extra = dict(statistics_dense_matmuls=stat_dense, statistics_projection=stat_proj)
    else:
        stat_dense = 0
    return dict(estimated_forward_flops=forward,estimated_training_flops=3*forward+(stat_dense if c.attention_kernel == 'original' else 0),
        components_forward=dict(backbone_projections=projections,vocab_head=head,
            attention=attention, **extra),
        accounting='FMA=2 FLOPs; allowed attention pairs, not masked dense-kernel execution. Backward approximated as2x consumed forward matrix FLOPs. Excludes softmax/norm/activation/optimizer/recomputation and is not measured hardware FLOPs.',
        statistics_note=('FLA fused backend shares QK/softmax; one extra P@phi(V) for variance or SiLU Jensen. Excludes nonlinear pointwise work and FlashAttention backward probability recomputation.' if c.attention_kernel == 'fla_triton' else 'Reference uses masked FP32 matrices and additional checkpoint recomputation; excludes pointwise operations.'),
        memory_note='Actual training peak VRAM must be measured separately.')
