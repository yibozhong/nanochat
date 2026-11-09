"""Analytic matrix FLOPs for vanilla and its optional residual/attention paths."""


def model_cost(config, batch=1, activation_bytes=2):
    c=config
    if c.architecture!='vanilla' or c.variant not in ('baseline','attnres','attnres_scale','skip_scale'):
        return None
    t,d,depth=c.sequence_length,c.width,c.depth
    projections=24*batch*t*depth*d*d
    head=2*batch*t*d*c.vocab_size
    attention=4*batch*d*depth*(t*(t+1)//2)
    if c.swa_looping:
        extra_passes=sum(passes-1 for _,passes in c.attention_schedule)
        loop_projections=8*batch*t*d*d*extra_passes
        attention=0
        for left,passes in c.attention_schedule:
            w=t if left<0 else min(t,left+1)
            pairs=t*w-w*(w-1)//2
            attention+=4*batch*d*pairs*passes
        forward=projections+loop_projections+head+attention
        return dict(estimated_forward_flops=forward,estimated_training_flops=3*forward,
            components_forward=dict(backbone_projections=projections,loop_projections=loop_projections,
                vocab_head=head,attention=attention),
            attention_schedule=list(c.attention_schedule),
            attention_evaluations=sum(p for _,p in c.attention_schedule),mlp_evaluations=depth,
            accounting='FMA=2; exact allowed causal/window pairs and all repeated QKVO projections. MLP once per layer. Backward approximately2x forward; excludes norm/softmax/activation/optimizer/recomputation and is not measured hardware FLOPs.')
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
    if c.variant in ('attnres','attnres_scale'):
        history=0;entries=0
        for i in range(depth):
            entries+=history+1
            if i%c.attnres_block_layers==0:history+=1
            entries+=history+1
        entries+=history+1
        scale_elements=(entries-1)*d if c.variant=='attnres_scale' else 0
        scale_forward=batch*t*scale_elements
        mixing=7*batch*t*d*entries+scale_forward
        return dict(estimated_forward_flops=forward+mixing,
            estimated_training_flops=3*(forward+mixing-scale_forward)+5*scale_forward-scale_elements,
            components_forward=dict(backbone_projections=projections,vocab_head=head,
                attention=attention,attnres_norm_score_mix=mixing,additional_value_scale=scale_forward),
            additional_scale_parameters=scale_elements,
            accounting='FMA=2; causal attention pairs; AttnRes approximated7 operations per entry/channel. Original backward approximated2x forward; extra value gate counts forward multiply, coefficient/gated input-gradient multiplies and deterministic gate-gradient multiply/reduction. Excludes scalar softmax/rsqrt, other norms/activations, optimizer, copies and recomputation; not measured hardware FLOPs.',
            final_history_state_bytes=batch*t*d*history*activation_bytes,
            memory_note='History alone excludes saved intermediate banks, gradients and allocator peak; measure full training peak.')
    return dict(estimated_forward_flops=forward,estimated_training_flops=3*forward,
        components_forward=dict(backbone_projections=projections,vocab_head=head,
            attention=attention),
        accounting='FMA=2 FLOPs; allowed attention pairs, not masked dense-kernel execution. Backward approximated as2x consumed forward matrix FLOPs. Excludes softmax/norm/activation/optimizer/recomputation and is not measured hardware FLOPs.',
        memory_note='Actual training peak VRAM must be measured separately.')
