"""Analytic matrix FLOPs for the retained full-attention vanilla model."""


def model_cost(config, batch=1, activation_bytes=2):
    c=config
    if c.architecture!='vanilla' or c.variant not in ('baseline','attnres','skip_scale'):
        return None
    t,d,depth=c.sequence_length,c.width,c.depth
    projections=24*batch*t*depth*d*d
    head=2*batch*t*d*c.vocab_size
    attention=4*batch*d*depth*(t*(t+1)//2)
    forward=projections+head+attention
    if c.variant=='skip_scale':
        # Relative to vanilla's already-present residual adds: one multiply
        # forward; dx multiply plus dscale multiply/reduction in backward.
        sites=2*depth
        extra_forward=sites*batch*t*d
        extra_train=4*extra_forward-sites*d
        return dict(estimated_forward_flops=forward+extra_forward,
            estimated_training_flops=3*forward+extra_train,
            components_forward=dict(backbone_projections=projections,vocab_head=head,
                attention=attention,additional_skip_scale=extra_forward),
            additional_scale_parameters=sites*d,
            accounting='FMA=2; causal attention pairs; backbone backward approximately2x forward. Scale extra FLOPs count forward multiply, dx multiply and dscale multiply/reduction. Excludes other elementwise work, optimizer, copies and recomputation; not measured hardware FLOPs.')
    if c.variant=='attnres':
        history=0;entries=0
        for i in range(depth):
            entries+=history+1
            if i%c.attnres_block_layers==0:history+=1
            entries+=history+1
        entries+=history+1
        mixing=7*batch*t*d*entries
        return dict(estimated_forward_flops=forward+mixing,
            estimated_training_flops=3*(forward+mixing),
            components_forward=dict(backbone_projections=projections,vocab_head=head,
                attention=attention,attnres_norm_score_mix=mixing),
            accounting='FMA=2; causal attention pairs; AttnRes approximated7 operations per entry/channel. Backward approximated2x forward. Excludes scalar softmax/rsqrt, other norms/activations, optimizer, copies and recomputation; not measured hardware FLOPs.',
            final_history_state_bytes=batch*t*d*history*activation_bytes,
            memory_note='History alone excludes saved intermediate banks, gradients and allocator peak; measure full training peak.')
    return dict(estimated_forward_flops=forward,estimated_training_flops=3*forward,
        components_forward=dict(backbone_projections=projections,vocab_head=head,
            attention=attention),
        accounting='FMA=2 FLOPs; allowed attention pairs, not masked dense-kernel execution. Backward approximated as2x consumed forward matrix FLOPs. Excludes softmax/norm/activation/optimizer/recomputation and is not measured hardware FLOPs.',
        memory_note='Actual training peak VRAM must be measured separately.')
