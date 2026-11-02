"""Analytic matrix FLOPs for vanilla and same-Transformer chunk reruns."""


def model_cost(config, batch=1, activation_bytes=2):
    c=config
    if c.architecture!='vanilla' or c.variant not in ('baseline','lookback_recompute'):
        return None
    t,d,depth=c.sequence_length,c.width,c.depth
    projections=24*batch*t*depth*d*d
    head=2*batch*t*d*c.vocab_size
    attention=4*batch*d*depth*(t*(t+1)//2)
    original=projections+head+attention
    reruns=consumed_reruns=bank_bytes=previous_hidden_bytes=0
    if c.variant=='lookback_recompute':
        bank_bytes=2*batch*t*d*depth*activation_bytes
        sizes=[min(c.recompute_chunk_size,t-i) for i in range(0,t,c.recompute_chunk_size)]
        if c.recompute_previous:
            previous_hidden_bytes=batch*min(c.recompute_chunk_size,t)*d*activation_bytes
            prefix=0
            for i,(previous,current) in enumerate(zip(sizes,sizes[1:])):
                pairs=previous*(prefix+current)+previous*(previous+1)//2
                block_flops=24*batch*previous*d*d+4*batch*d*pairs
                reruns+=depth*block_flops
                # Last boundary has no subsequent NTP consumer. Earlier reruns
                # consume the last layer's K/V, but not its output/MLP branch.
                if i<len(sizes)-2:
                    consumed_reruns+=(depth-1)*block_flops+4*batch*previous*d*d
                prefix+=previous
    forward=original+reruns
    return dict(estimated_forward_flops=forward,estimated_training_flops=3*original+reruns+2*consumed_reruns,
        components_forward=dict(backbone_projections=projections,vocab_head=head,
            attention=attention,memory_projection=0,lookback=reruns),
        logical_memory_kv_bytes=bank_bytes,logical_previous_hidden_bytes=previous_hidden_bytes,
        accounting='FMA=2 FLOPs; allowed attention pairs, not masked dense-kernel execution. Includes retrospective chunk reruns; backward approximated as2x consumed forward matrix FLOPs. Excludes softmax/norm/activation/optimizer/activation-checkpoint recomputation and is not measured hardware FLOPs.',
        memory_note='Actual training peak VRAM must be measured separately.')
