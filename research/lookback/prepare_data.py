"""Pack fresh16K FineWeb sequences for the matched Lookback10B experiment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path('nanochat-src').resolve()))
os.environ['NANOCHAT_BASE_DIR'] = str(Path('cache').resolve())
import numpy as np
import torch
import nanochat.dataset as dataset
from nanochat.tokenizer import get_tokenizer
from nanochat.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    dataset.DATA_DIR=str(Path('raw').resolve())
    T,B,batch_tokens=16384,2,32768
    parser=argparse.ArgumentParser()
    parser.add_argument('--training-steps',type=int,default=305176)
    parser.add_argument('--first-train-shard-only',action='store_true')
    args=parser.parse_args()
    steps=args.training_steps
    assert steps>=128
    requested=10_000_000_000 if steps==305176 else steps*batch_tokens
    tokenizer=get_tokenizer();assert tokenizer.get_vocab_size()<=65536
    manifest=json.loads(Path('dataset-manifest.json').read_text())
    provenance=json.loads(Path('cache/tokenizer-provenance.json').read_text())
    Path('data').mkdir(exist_ok=True);Path('results').mkdir(exist_ok=True)
    start=time.perf_counter()
    selected_files=manifest['files']
    if args.first_train_shard_only:
        assert steps==128
        selected_files=[selected_files[0],selected_files[-1]]
    for item in selected_files:
        path=Path('raw',item['filename'])
        assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],path
    print(f'All {len(selected_files)} selected raw shard SHA-256 checks passed',flush=True)
    assert T*B==batch_tokens
    records={};epochs=[];prefix_sha=None
    for split,paths,counts in [('train',['train'],[steps]),('val',['dev','test'],[4,64])]:
        loader=tokenizing_distributed_data_loader_with_state_bos_bestfit(tokenizer,B,T,split,device='cpu')
        for name,count in zip(paths,counts):
            path=Path('data',name+'.bin');digest=hashlib.sha256()
            with path.open('wb') as f:
                for i in range(count):
                    x,y,state=next(loader)
                    rows=torch.cat((x,y[:,-1:]),dim=1).numpy().astype(np.uint16)
                    assert np.array_equal(rows[:,1:],y.numpy())
                    content=rows.tobytes();f.write(content);digest.update(content)
                    if name=='train':
                        if not epochs or epochs[-1]['source_state']['epoch']!=state['epoch']:
                            epochs.append(dict(batch=i+1,source_state=dict(state)))
                        if i+1==128:
                            prefix_sha=digest.hexdigest()
                    if i%500==0 or i==count-1:
                        row=dict(split=name,batches=i+1,tokens=(i+1)*batch_tokens,
                                 seconds=time.perf_counter()-start,source_state=state)
                        print(json.dumps(row),flush=True)
                        Path('results/progress.json').write_text(json.dumps(row,indent=2)+'\n')
            assert sha(path)==digest.hexdigest(),'Readback hash mismatch'
            records[name]=dict(rows=count*B,tokens=count*batch_tokens,bytes=path.stat().st_size,
                               sha256=digest.hexdigest(),last_source_state=dict(state))
    info=dict(complete=True,logging_backend='local',sequence_length=T,dtype='uint16',
        layout='rows of16385 token ids; x=row[:-1], y=row[1:]',batch_tokens=batch_tokens,
        training_steps=steps,requested_tokens=requested,training_tokens=steps*batch_tokens,
        files=records,tokenizer=provenance,native_commit='145da40cdd37b9522aa53d26c0c356e9126458bb',
        dataset_revision='87f09149ef4734204d70ed1d046ddc9ca3f2b8f9',train_shards='000 only (validation prefix)' if args.first_train_shard_only else '000..012',heldout_shard='013',
        first_128_steps_sha256=prefix_sha,source_read_epochs=epochs,
        epoch_note='Native loader cycles the entire 13-shard training split. Epoch tracks source reads into the document buffer, not exact consumed unique tokens.',
        seconds=time.perf_counter()-start,validation='dev first8 rows/131072tokens; test next128 rows/2097152tokens; disjoint packed rows and same heldout shard013; not the old2K dev/test examples')
    for p in ('data/manifest.json','results/data.json'):
        Path(p).write_text(json.dumps(info,indent=2)+'\n')
    # Small exact-prefix cache for GPU checks and25->50 fresh-process resume.
    import shutil
    Path('gate-data/data').mkdir(parents=True,exist_ok=True)
    prefix_bytes=128*B*(T+1)*2
    with Path('data/train.bin').open('rb') as src,Path('gate-data/data/train.bin').open('wb') as dst:
        dst.write(src.read(prefix_bytes))
    assert sha(Path('gate-data/data/train.bin'))==prefix_sha
    for split in ('dev','test'):shutil.copy2(Path('data')/(split+'.bin'),Path('gate-data/data')/(split+'.bin'))
    gate=json.loads(json.dumps(info))
    gate.update(training_steps=128,training_tokens=128*batch_tokens,requested_tokens=128*batch_tokens,purpose='Exact16K prefix for correctness and restart checks only')
    gate['files']['train'].update(rows=128*B,tokens=128*batch_tokens,bytes=prefix_bytes,sha256=prefix_sha)
    Path('gate-data/data/manifest.json').write_text(json.dumps(gate,indent=2)+'\n')


if __name__=='__main__':main()
