"""Worker-only extraction of the unchanged packed-data prefix for short tests."""
import copy
import json
from pathlib import Path
import tarfile

out=Path('data');out.mkdir(exist_ok=True)
with tarfile.open('nanochat-2k-10b-data-7685535.tar') as archive:
    original=json.load(archive.extractfile('data/manifest.json'))
    manifest=copy.deepcopy(original)
    Path('results/source-data-manifest.json').write_text(json.dumps(original,indent=2)+'\n')
    for split,rows in (('train',2048*16),('dev',64),('test',1024)):
        size=rows*(original['sequence_length']+1)*2
        assert rows<=original['files'][split]['rows']
        source=archive.extractfile('data/'+split+'.bin')
        with (out/(split+'.bin')).open('wb') as dest:
            remaining=size
            while remaining:
                chunk=source.read(min(remaining,1024**2));assert chunk
                dest.write(chunk);remaining-=len(chunk)
        manifest['files'][split]=dict(rows=rows,tokens=rows*original['sequence_length'],bytes=size)
    manifest.update(training_steps=2048,training_tokens=2048*32768,requested_tokens=2048*32768,
        purpose='debug prefix, same packed rows as the original formal dataset; no repacking',
        source_uri='osdf:///chtc/staging/y/yzhong239/nanochat-2k-10b-data-7685535.tar')
    # Whole-source content digests do not describe this prefix.
    manifest.pop('first_1b_prefix_sha256',None)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
