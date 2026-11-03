"""CPU-only packing with the native document order, ties, cropping and refills.

This is an offline preparation adapter, not a replacement for the training
dataloader. Full-cache hashes must still match the original implementation.
"""
from bisect import bisect_left, bisect_right, insort
from collections import deque

import numpy as np
import pyarrow.parquet as pq
import torch

from nanochat.dataset import list_parquet_files


class LengthBuffer:
    def __init__(self):
        self.lengths = []
        self.queues = {}
        self.count = 0

    def add(self, tokens):
        size = len(tokens)
        assert size > 0, 'BOS-prefixed documents are nonempty'
        if size not in self.queues:
            insort(self.lengths, size)
            self.queues[size] = deque()
        self.queues[size].append(tokens)
        self.count += 1

    def take(self, remaining):
        index = bisect_right(self.lengths, remaining) - 1
        # Native code picks the first shortest document if none fits.
        size = self.lengths[max(index, 0)]
        queue = self.queues[size]
        tokens = queue.popleft()  # Preserve earliest surviving insertion on ties.
        if not queue:
            del self.queues[size]
            self.lengths.pop(bisect_left(self.lengths, size))
        self.count -= 1
        return tokens


def text_batches(split, batch_size):
    paths = list_parquet_files(warn_on_legacy=split == 'train')
    paths = paths[:-1] if split == 'train' else paths[-1:]
    assert paths
    epoch = 1
    while True:
        for pq_idx, path in enumerate(paths):
            pf = pq.ParquetFile(path)
            for rg_idx in range(pf.num_row_groups):
                texts = pf.read_row_group(rg_idx, columns=['text']).column('text').to_pylist()
                # Do not merge across row groups or change tokenizer batch size:
                # these boundaries determine native buffer refill order.
                for start in range(0, len(texts), batch_size):
                    yield texts[start:start + batch_size], (pq_idx, rg_idx, epoch)
        epoch += 1


def loader(tokenizer, B, T, split, tokenizer_threads=4, tokenizer_batch_size=128,
           device='cpu', resume_state_dict=None, buffer_size=1000):
    assert device == 'cpu' and resume_state_dict is None
    assert split in ('train', 'val')
    assert B > 0 and T > 0 and buffer_size > 0
    batches = text_batches(split, tokenizer_batch_size)
    buffer = LengthBuffer()
    bos = tokenizer.get_bos_token_id()
    # One reusable contiguous row buffer; tensor views replace the native
    # row->inputs/targets->device-buffer copies in this CPU-only use case.
    rows = np.empty((B, T + 1), dtype=np.int64)
    view = torch.from_numpy(rows)
    while True:
        for row in rows:
            pos = 0
            while pos < T + 1:
                while buffer.count < buffer_size:
                    texts, (pq_idx, rg_idx, epoch) = next(batches)
                    for tokens in tokenizer.encode(texts, prepend=bos, num_threads=tokenizer_threads):
                        buffer.add(tokens)
                remaining = T + 1 - pos
                tokens = buffer.take(remaining)
                take = min(remaining, len(tokens))
                row[pos:pos + take] = tokens[:take]
                pos += take
        yield view[:, :-1], view[:, 1:], dict(pq_idx=pq_idx, rg_idx=rg_idx, epoch=epoch)
