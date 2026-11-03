"""Check regenerated 2K data against the original splits and save a small prefix."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    actual = json.loads(Path('data/manifest.json').read_text())
    expected = json.loads(Path('expected-data.json').read_text())
    for key in ('sequence_length', 'training_steps', 'training_tokens', 'tokenizer',
                'dataset_revision', 'native_commit', 'first_1b_prefix_sha256'):
        assert actual[key] == expected[key], key
    assert actual['sequence_length'] == 2048 and actual['complete']
    for split in ('train', 'dev', 'test'):
        for key in ('rows', 'tokens', 'bytes', 'sha256'):
            assert actual['files'][split][key] == expected['files'][split][key], (split, key)
        # prepare_data.py already rehashed each file after writing it.
        assert Path('data', split + '.bin').stat().st_size == actual['files'][split]['bytes']

    dest = Path('gate-data/data')
    dest.mkdir(parents=True, exist_ok=True)
    steps, rows = 128, 128 * 16
    prefix_bytes = rows * 2049 * 2
    with Path('data/train.bin').open('rb') as src, (dest / 'train.bin').open('wb') as dst:
        prefix = src.read(prefix_bytes)
        assert len(prefix) == prefix_bytes
        dst.write(prefix)
    for split in ('dev', 'test'):
        shutil.copy2(Path('data', split + '.bin'), dest / (split + '.bin'))
    gate = json.loads(json.dumps(actual))
    gate.update(training_steps=steps, training_tokens=steps * 32768,
                requested_tokens=steps * 32768,
                purpose='Exact 2K training prefix for checks; not a separate training run')
    gate['files']['train'].update(rows=rows, tokens=steps * 32768, bytes=prefix_bytes,
                                sha256=sha(dest / 'train.bin'))
    gate['files']['train'].pop('last_source_state', None)
    gate.pop('source_read_epochs', None)
    gate.pop('first_1b_prefix_sha256', None)
    (dest / 'manifest.json').write_text(json.dumps(gate, indent=2) + '\n')
    with tarfile.open('gate-data.tar', 'w') as archive:
        archive.add(dest, arcname='data')
    result = dict(complete=True, logging_backend='local', sequence_length=2048,
                  original_split_hashes_matched=True, training_tokens=actual['training_tokens'],
                  split_sha256={k: v['sha256'] for k, v in actual['files'].items()},
                  gate_manifest=gate, gate_archive=dict(bytes=Path('gate-data.tar').stat().st_size,
                                                       sha256=sha(Path('gate-data.tar'))))
    Path('results/restoration.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
