"""Run the unchanged cache writer with the verified offline packing adapter."""
import functools
import os
from pathlib import Path
import runpy
import sys

sys.path.insert(0, str(Path('nanochat-src').resolve()))
os.environ['NANOCHAT_BASE_DIR'] = str(Path('cache').resolve())
import nanochat.dataloader as dataloader
from fast_pack import loader

threads = int(os.environ.get('NANOCHAT_PACKING_THREADS', '4'))
assert threads in (4, 8)
dataloader.tokenizing_distributed_data_loader_with_state_bos_bestfit = functools.partial(
    loader, tokenizer_threads=threads)
runpy.run_path('prepare_native.py', run_name='__main__')
