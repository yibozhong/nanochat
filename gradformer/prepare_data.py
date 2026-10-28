"""Create fixed TinyStories V2 train/validation token streams in a CPU job."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import requests
import sentencepiece as spm

_tokenizer = None


def init_worker(path):
    global _tokenizer
    _tokenizer = spm.SentencePieceProcessor(model_file=path)


def encode(story):
    return [_tokenizer.bos_id()] + _tokenizer.encode(story) + [_tokenizer.eos_id()]


def stories(path):
    text = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip() == "<|endoftext|>":
                if text:
                    yield "".join(text).strip()
                    text = []
            else:
                text.append(line)
    if text:
        yield "".join(text).strip()


def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--output", default="dataset")
    p.add_argument("--train-tokens", type=int, default=300_000_000)
    p.add_argument("--val-tokens", type=int, default=2_000_000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--revision", default="f54c09fd23315a6f9c86f9dc80f725de7d8f9c64",
                   help="Pinned Hugging Face dataset commit")
    args = p.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    revision = args.revision
    tokenizer = spm.SentencePieceProcessor(model_file=args.tokenizer)
    assert tokenizer.vocab_size() < 65536
    import wandb
    run = wandb.init(project="nanochat", entity="yibozhong657-none", job_type="data",
                     name="gradformer-tinystories-v2-prepare", config=vars(args))
    metadata = dict(dataset="roneneldan/TinyStories", revision=revision, variant="V2-GPT4",
                    tokenizer_sha256=digest(args.tokenizer), vocab_size=tokenizer.vocab_size(),
                    dtype="uint16", separators="BOS + story + EOS", splits={})
    # Use bounded chunks so ProcessPoolExecutor cannot queue the whole corpus.
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                             initargs=(args.tokenizer,)) as pool:
        for split, limit in (("train", args.train_tokens), ("valid", args.val_tokens)):
            filename = f"TinyStoriesV2-GPT4-{split}.txt"
            source = out / filename
            response = requests.get(f"https://huggingface.co/datasets/roneneldan/TinyStories/resolve/{revision}/{filename}",
                                    stream=True, timeout=(30, 300))
            response.raise_for_status()
            with open(source, "wb") as f:
                for chunk in response.iter_content(8 * 1024**2):
                    f.write(chunk)
            name = "val" if split == "valid" else split
            tokens_path = out / (name + ".bin")
            count, documents = 0, 0
            iterator = iter(stories(source))
            with open(tokens_path, "wb") as f:
                while count < limit:
                    chunk = []
                    for _ in range(512):
                        item = next(iterator, None)
                        if item is None:
                            break
                        chunk.append(item)
                    if not chunk:
                        break
                    for ids in pool.map(encode, chunk, chunksize=16):
                        # Whole-document truncation preserves explicit boundaries.
                        array = np.asarray(ids, dtype=np.uint16)
                        f.write(array.tobytes()); count += len(ids); documents += 1
                        if count >= limit:
                            break
                    if documents % 51200 == 0:
                        print(name, documents, count, flush=True)
            metadata["splits"][name] = dict(tokens=count, documents=documents,
                                            sha256=digest(tokens_path), source_sha256=digest(source))
            run.log({f"data/{name}_tokens": count, f"data/{name}_documents": documents})
            source.unlink()  # Derived from the pinned remote source; only token streams are archived.
    with open(out / "manifest.json", "w") as f:
        json.dump(metadata, f, indent=2)
    run.summary.update(metadata)
    run.finish()


if __name__ == "__main__":
    main()
