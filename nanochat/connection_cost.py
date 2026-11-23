"""Analytic matrix FLOPs for the retained full-attention vanilla model."""


def model_cost(config, batch=1, activation_bytes=2, feedback_passes=None):
    c=config
    if c.architecture!='vanilla' or c.variant not in ('baseline','attnres','attnres_scale','skip_scale'):
        return None
    t,d,depth=c.sequence_length,c.width,c.depth
    projections=2*batch*t*depth*(4*d*d + 2*d*c.ffn_width)
    head=2*batch*t*d*c.vocab_size
    attention=4*batch*d*depth*(t*(t+1)//2)
    forward=projections+head+attention
    if c.residual_mode != 'vanilla':
        # Matrix-only reference plus residual multiply/add/reduction arithmetic.
        # Normalized mode: target norm (3d-1), target-h (d), HB (5d), hidden norm (3d-1).
        # Coefficient scalar work and sqrt/division are excluded here.
        sites=2*depth
        per_site={'heavy_ball':5*d, 'heavy_ball_rms':8*d-1, 'ngpt_heavy_ball':12*d-2, 'ngpt_hbformer':25*d-6, 'ngpt':9*d-2, 'ngpt_lookahead':29*d-6, 'lookahead_only':7*d}[c.residual_mode]
        initial=3*d-1
        if c.ngpt_velocity_mode == 'lookahead_geodesic_split':
            # Leading vector arithmetic per token: read 5d, sphere log ~8d,
            # split + EMA 11d, Exp/transport + roundoff projections ~19d.
            per_site=43*d
        if c.residual_mode == 'ngpt_lookahead':
            initial += 4*d-1  # v0 tangent projection
        if c.residual_mode == 'heavy_ball_rms' and c.velocity_init == 'token':
            initial += 3*d-1
        if c.residual_mode == 'ngpt_heavy_ball' and c.ngpt_velocity_mode == 'transport_rms':
            # Target norm + projected mixture + speed norm + retraction + transport.
            per_site=25*d-6
            if c.velocity_init == 'token':
                initial += 7*d-2  # initial embedding tangent projection and RMS norm
        gain_parameters = 0
        if c.ngpt_velocity_mode == 'transport_layernorm':
            # Centering (mean + subtract) and per-channel affine gain vs. RMS.
            per_site += 3*d-1
            gain_parameters = d*sites
        residual=batch*t*(sites*per_site+initial)
        embedding_parameters=c.vocab_size*d if c.velocity_init=='token' else 0
        if c.velocity_init == 'token_position':
            residual += batch*t*d  # v0 = token + position
            embedding_parameters = (c.vocab_size+c.sequence_length)*d
        state_bytes = {'ngpt':4, 'lookahead_only':activation_bytes+4}.get(c.residual_mode, 8)
        if c.chord_state_dtype == 'bfloat16':
            state_bytes = 4  # Two BF16 streams per token and channel.
        scalars_per_site = {'ngpt':1, 'ngpt_hbformer':2, 'ngpt_lookahead':2, 'lookahead_only':1}.get(c.residual_mode, 3)
        if c.ngpt_velocity_mode == 'lookahead_geodesic_split':
            scalars_per_site += 1
        if c.ngpt_velocity_mode == 'lookahead_axis_write' or c.split_chord_write:
            # Two dot products, multiply the projection by its coefficient,
            # then scale v and add. Mean/division scalar work is excluded.
            residual += batch*t*sites*(6*d-1)
            if c.split_chord_write:
                residual += sites  # parallel minus perpendicular coefficient, shared across tokens
                if c.signed_chord_steps:
                    residual += 6*sites  # two scalar tangent extensions, shared across tokens
            scalars_per_site += 1
        if c.ngpt_velocity_mode == 'lookahead_split_learned_beta':
            gain_parameters += sites - 1  # Final FFN velocity does not reach the logits.
            residual += sites - 1  # 1 - beta, shared across tokens; sigmoid excluded.
        if c.chord_query_only_lookahead:
            residual += batch*t*depth*(3*d-1)  # Additional shared RMSNorm for K/V.
        if c.chord_velocity_write:
            gain_parameters += sites
            residual += batch*t*sites  # One scalar add per token in the existing velocity coefficient.
        if c.channel_chord_coefficients:
            gain_parameters += (3*sites - 1)*(d - 1)
            # Both (parallel - perpendicular) and (1 - beta) are now vectors.
            residual += (2*sites - 1)*(d - 1)
            # (parallel - perpendicular) * dot(delta, v) / dot(v, v) per token.
            residual += batch*t*sites*(d - 1)
        if c.token_chord_gates:
            gain_parameters += 2*sites
            # RMS(delta), cosine and per-token gate scalings; reuse dot(delta,v) and RMS(v).
            residual += batch*t*sites*(2*d+9) - sites
        if c.token_beta_gate:
            gain_parameters += sites - 1
            # Project the realized chord, compute the two RMS norms and their dot product.
            residual += (sites-1)*(batch*t*(10*d+2)-1)
        history = 0
        depth_mix = 0
        if c.chord_attnres:
            stride = c.attnres_block_layers
            history = (depth - 1) // stride
            if c.chord_history_correction:
                mixed_sites = 2*max(0, depth-stride)
                # A single endpoint needs a learned gain but no score parameters or softmax.
                gain_parameters += mixed_sites + 4*d*max(0, depth-2*stride)
                scored_entries = sum(2*(i//stride) for i in range(2*stride, depth))
                depth_mix = batch*t*d*(7*scored_entries + 2*mixed_sites)
                if c.chord_history_updates:
                    depth_mix += batch*t*d*history  # One subtraction per stored coordinate.
            else:
                gain_parameters += 4*d*max(0, depth-stride)
                entries = sum(2*(1+i//stride) for i in range(stride, depth))
                depth_mix = 7*batch*t*d*entries
            residual += depth_mix
        # Reconstruct the attention and FFN output projections once per layer.
        remat_flops = 2*batch*t*depth*d*(d+c.ffn_width) if c.recompute_chord_force else 0
        extra_ffn_params = 2*depth*d*(c.ffn_width - 4*d)
        return dict(estimated_forward_flops=forward+residual,
            estimated_training_flops=3*(forward+residual)+remat_flops,
            force_recomputation_flops=remat_flops,
            components_forward=dict(backbone_projections=projections,vocab_head=head,
                attention=attention,residual_arithmetic=residual),
            additional_parameters=scalars_per_site*sites+embedding_parameters+gain_parameters+extra_ffn_params,
            **(dict(depth_attention_forward_flops=depth_mix, endpoint_history_bytes=history*batch*t*d*activation_bytes)
               if c.chord_attnres else {}),
            ffn_intermediate_width=c.ffn_width, additional_ffn_parameters=extra_ffn_params,
            velocity_embedding_parameters=embedding_parameters,
            live_residual_state_bytes=state_bytes*batch*t*d,
            accounting='FMA=2; causal attention pairs; residual arithmetic estimate and backward approximated as 2x forward. Scalar transcendentals, division, optimizer (including post-step weight normalization), copies and checkpoint recomputation excluded. Not hardware-measured FLOPs.',
            memory_note='Live hidden and velocity where used. Additive lookahead uses a backbone-dtype hidden and FP32 velocity. Autograd intermediates, embedding/optimizer bandwidth and allocator peak must be measured.')
    if c.latent_feedback:
        backbone=projections+attention
        fusion=2*batch*t*d*d
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
