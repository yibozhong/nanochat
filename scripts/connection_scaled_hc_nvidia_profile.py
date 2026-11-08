"""Same full-step harness, selecting only the isolated scaled HC implementation."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys


def legacy_function():
    path=Path(__file__).resolve().parents[1]/'research/connections/scaled-hc-nvidia/legacy_mixing.py'
    spec=importlib.util.spec_from_file_location('scaled_hc_legacy_benchmark',path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module.scaled_mix


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--implementation',choices=('legacy','nvidia','shared'),required=True)
    opts,rest=parser.parse_known_args()
    if opts.implementation=='legacy':
        import nanochat.scaled_connections as connections
        connections.scaled_mix=legacy_function()
    sys.argv=[sys.argv[0],*rest]
    from scripts.connection_context_profile import main as profile_main
    profile_main()
    path=Path(rest[rest.index('--output')+1])
    value=json.loads(path.read_text());value['implementation']=opts.implementation
    path.write_text(json.dumps(value,indent=2)+'\n')


if __name__=='__main__':main()
