"""Optional-import decorator shim for the vendored NVIDIA AttnRes kernel."""


def null_decorator(*args, **kwargs):
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]
    return lambda fn: fn
