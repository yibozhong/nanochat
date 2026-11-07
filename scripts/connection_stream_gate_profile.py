"""Profile either implementation of the same independent-stream sigmoid HC."""
import argparse
import sys
import nanochat.stream_gated_connections as connections
import nanochat.post_stream_gated_connections as post_connections
from nanochat.stream_gate_mixing import compiled_stream_gate_reference
from nanochat.post_stream_gate_mixing import compiled_post_stream_gate_reference


def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--gate-backend',choices=('fused','torch'),default='fused')
    a,rest=p.parse_known_args()
    if a.gate_backend=='torch':
        connections.stream_gate_mix=compiled_stream_gate_reference
        post_connections.post_stream_gate_mix=compiled_post_stream_gate_reference
    sys.argv=[sys.argv[0],*rest]
    from scripts.connection_context_profile import main as profile_main
    profile_main()


if __name__=='__main__':main()
