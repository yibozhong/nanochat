"""Analytic matrix FLOPs for the retained full-attention vanilla model."""


def model_cost(config, batch=1, activation_bytes=2, feedback_passes=None):
    c=config
    if c.architecture!='vanilla' or c.variant not in ('baseline','attnres','attnres_scale','skip_scale'):
        return None
    t,d,depth=c.sequence_length,c.width,c.depth
    projections=24*batch*t*depth*d*d
    head=2*batch*t*d*c.vocab_size
    attention=4*batch*d*depth*(t*(t+1)//2)
    forward=projections+head+attention
    if c.latent_feedback:
        backbone=projections+attention
        fusion=2*batch*t*d*d
        # P1/P2 mixture expectation unless accounting for a particular update.
        p=1.85 if feedback_passes is None else feedback_passes
        if p<1:
            raise ValueError('Feedback passes must be >=1')
        extra=p-1
        fusion_train=2*extra if feedback_passes is None else (p if p>1 else 0)
        return dict(estimated_forward_flops=p*backbone+head+extra*fusion,
            estimated_training_flops=(p+2)*backbone+3*head+fusion_train*fusion,
            estimated_evaluation_flops=c.feedback_eval_passes*backbone+head+(c.feedback_eval_passes-1)*fusion,
            additional_parameters=d*d, expected_training_passes=p,
            accounting='FMA=2; only final backbone pass and head have backward. Final fusion has weight gradient only (detached input); earlier fusion passes have no backward. Default averages 15% P1 / 85% P2. Excludes elementwise work, optimizer and checkpoint recomputation.')
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
