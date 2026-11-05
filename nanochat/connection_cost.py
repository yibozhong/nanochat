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
    result = dict(estimated_forward_flops=forward,estimated_training_flops=3*forward,
        components_forward=dict(backbone_projections=projections,vocab_head=head,
            attention=attention),
        accounting='FMA=2 FLOPs; allowed attention pairs, not masked dense-kernel execution. Backward approximated as2x consumed forward matrix FLOPs. Excludes softmax/norm/activation/optimizer/recomputation and is not measured hardware FLOPs.',
        memory_note='Actual training peak VRAM must be measured separately.')
    if c.ffn_width_residual:
        result['ffn_width_residual'] = dict(additional_parameters=0, additional_matrix_flops=0,
            live_state_bytes=batch*t*(4*d)*activation_bytes,
            note='Extra RMS reductions, elementwise operations and saved backward activations are not included in the matrix FLOPs estimate.')
    return result
