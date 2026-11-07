"""Select an audited post-gate implementation for a fresh-process benchmark."""
import argparse
import importlib.util
from pathlib import Path
import sys


def legacy_module():
    path=Path(__file__).resolve().parents[1]/'research/connections/post-gate-opt/reference/post_stream_gate_mixing.py'
    name='post_gate_validated_legacy'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return sys.modules[name]


def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--implementation',choices=('pre','legacy','optimized','torch'),required=True)
    a,rest=p.parse_known_args()
    import nanochat.post_stream_gated_connections as connections
    from nanochat.post_stream_gate_mixing import compiled_post_stream_gate_reference
    if a.implementation=='legacy':connections.post_stream_gate_mix=legacy_module().post_stream_gate_mix
    elif a.implementation=='torch':connections.post_stream_gate_mix=compiled_post_stream_gate_reference
    sys.argv=[sys.argv[0],*rest]
    from scripts.connection_context_profile import main as profile_main
    profile_main()


if __name__=='__main__':main()
