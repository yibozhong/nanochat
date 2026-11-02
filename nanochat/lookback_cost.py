"""Analytic matrix-multiply FLOPs; not hardware-counter measurements."""


def model_cost(config, batch=1, activation_bytes=2):
    c = config
    if c.architecture != 'vanilla' or c.variant not in ('baseline', 'lookback'):
        return None
    t, d, depth = c.sequence_length, c.width, c.depth
    projections = 24*batch*t*depth*d*d
    head = 2*batch*t*d*c.vocab_size
    bank = unused_bank = update = unused_update = 0
    if c.variant == 'baseline':
        pairs = t*(t+1)//2
        bank_bytes = previous_bytes = 0
    else:
        sizes = [min(c.chunk_size, t-i) for i in range(0, t, c.chunk_size)]
        pairs = 0; history = 0
        for size in sizes:
            pairs += size*history+size*(size+1)//2
            history += size
        bank = 4*batch*t*d*d
        unused_bank = 4*batch*sizes[-1]*d*d
        if c.lookback_enabled:
            for previous, current in zip(sizes, sizes[1:]):
                boundary = 4*batch*(previous+current)*d*d+4*batch*d*previous*current
                update += boundary
                bank += 4*batch*previous*d*d
            if len(sizes)>1:
                unused_update = boundary
                unused_bank += 4*batch*sizes[-2]*d*d
        bank_bytes = 2*batch*t*d*activation_bytes
        previous_bytes = batch*sizes[-1]*d*activation_bytes
    attention = 4*batch*d*depth*pairs
    forward = projections+head+attention+bank+update
    # The last write/revision is executed exactly once but has no later NTP consumer.
    training = 3*forward-2*(unused_bank+unused_update)
    return dict(estimated_forward_flops=forward, estimated_training_flops=training,
        components_forward=dict(backbone_projections=projections, vocab_head=head,
            attention=attention, memory_projection=bank, lookback=update),
        logical_memory_kv_bytes=bank_bytes, logical_previous_hidden_bytes=previous_bytes,
        accounting='FMA=2 FLOPs; allowed attention pairs, not masked dense-kernel execution. Backward approximated as2x consumed forward matrix FLOPs. Excludes softmax/norm/activation/optimizer/recomputation and is not measured hardware FLOPs.',
        memory_note='Logical live bank only. Autograd retains earlier versions until backward; this does not imply an equal reduction in training peak VRAM.')
