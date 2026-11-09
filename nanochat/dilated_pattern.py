"""Geometry of causal sink + local + progressively dilated attention."""


def dilation_schedule(context, depth, budget=256, core=128, sinks=4, progressive=True):
    if any(type(x) is not int for x in (context, depth, budget, core, sinks)):
        raise ValueError('Attention geometry must use integers')
    if context < 1 or depth < 2 or core < 1 or not 1 <= sinks <= 16 or budget-core-sinks < 2:
        raise ValueError('Require context>=1, depth>=2, core>=1,1<=sinks<=16 and remote>=2')
    remote=budget-core-sinks
    strides=[]
    for layer in range(depth-1):
        target=max(budget,(context*(layer+1)+depth-1)//depth)
        numerator=max(0,target-1-core);denominator=remote-1
        # Nearest integer, ties upward. The final full layer has no stride.
        stride=max(1,(2*numerator+denominator)//(2*denominator)) if progressive else 1
        strides.append(stride)
    return tuple(strides)+(None,)


def selected_positions(query, stride, core, remote, sinks):
    """Small independent set definition for inspection/tests, not GPU execution."""
    if query < 0:raise ValueError('Query position must be nonnegative')
    positions=set(range(min(sinks,query+1)))
    positions.update(range(max(0,query-core+1),query+1))
    positions.update(query-core-i*stride for i in range(remote) if query-core-i*stride>=sinks)
    return sorted(positions)
