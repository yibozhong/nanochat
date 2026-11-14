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
        if c.feedback_source in ('embedding_qkv','per_layer_qkv'):
            readers=depth if c.feedback_source=='per_layer_qkv' else 1
            # Q/O have input and weight gradients; detached bank K/V only
            # need weight gradients. Bank normalization is shared by readers.
            bank_norm=4*batch*t*depth*d
            read_forward=4*batch*t*depth*d
            reassociated=c.feedback_projection_order=='reassociated'
            read_backward=(6 if reassociated else 10)*batch*t*depth*d
            forward_gemms=4 if reassociated else 2+2*depth
            train_gemms=12 if reassociated else 6+4*depth
            mix_forward=bank_norm+readers*(forward_gemms*fusion+read_forward)
            mix_train=bank_norm+readers*(train_gemms*fusion+read_forward+read_backward)
            p=1.85 if feedback_passes is None else feedback_passes
            if p<1:raise ValueError('Feedback passes must be >=1')
            extra=.85*mix_train if feedback_passes is None else ((p-2)*mix_forward+mix_train if p>1 else 0)
            return dict(estimated_forward_flops=p*backbone+head+(p-1)*mix_forward,
                estimated_training_flops=(p+2)*backbone+3*head+extra,
                estimated_evaluation_flops=c.feedback_eval_passes*backbone+head+(c.feedback_eval_passes-1)*mix_forward,
                additional_parameters=4*readers*d*d,expected_training_passes=p,
                detached_layer_bank_bytes=batch*t*depth*d*activation_bytes,
                projected_kv_saved_bytes=0 if reassociated else 2*readers*batch*t*depth*d*activation_bytes,
                projection_order=c.feedback_projection_order,
                accounting='FMA=2; independent full-width Q/K/V/O per reader; bank detached, all projection weights learn. Reassociated order applies K to Q and V after mixing: four tokenwise forward GEMMs, all with input/weight gradients. Materialized order projects every source layer. Default averages 15% P1 / 85% P2; final-only backbone/head backward. Excludes scalar softmax/rsqrt, other elementwise work, optimizer and recomputation.',
                memory_note='Bank estimates exclude model, gradients, backward temporaries and allocator peak; measure full training peak.')
        if c.feedback_source=='layers':
            # Query/output GEMMs both backpropagate to their inputs. Bank is
            # detached. Read: norm, scores, depth mix; backward only computes dQ.
            read_forward=8*batch*t*depth*d
            read_backward=6*batch*t*depth*d
            mix_forward=2*fusion+read_forward
            mix_train=6*fusion+read_forward+read_backward
            p=1.85 if feedback_passes is None else feedback_passes
            if p<1:
                raise ValueError('Feedback passes must be >=1')
            extra_mix_train=.85*mix_train if feedback_passes is None else ((p-2)*mix_forward+mix_train if p>1 else 0)
            return dict(estimated_forward_flops=p*backbone+head+(p-1)*mix_forward,
                estimated_training_flops=(p+2)*backbone+3*head+extra_mix_train,
                estimated_evaluation_flops=c.feedback_eval_passes*backbone+head+(c.feedback_eval_passes-1)*mix_forward,
                additional_parameters=2*d*d,expected_training_passes=p,
                detached_layer_bank_bytes=batch*t*depth*d*activation_bytes,
                accounting='FMA=2; detached preliminary passes, final-only head/backbone backward; full-width query/output projections, no K/V projections. Read approximates reductions and excludes softmax/rsqrt scalar costs, other elementwise work, optimizer and recomputation.')
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
