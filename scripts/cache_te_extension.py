"""Preserve the built official C++ extension for retries; no source patching."""
import base64
import csv
import hashlib
from importlib.metadata import distribution
import io
import json
from pathlib import Path
import zipfile


def main():
    dist = distribution('transformer-engine-torch')
    assert dist.version == '2.20.2'
    wheel = Path('results/transformer_engine_torch-2.20.2-cp311-cp311-linux_x86_64.whl')
    rows = []
    with zipfile.ZipFile(wheel, 'w', zipfile.ZIP_DEFLATED) as out:
        for entry in dist.files:
            name = str(entry)
            assert not entry.is_absolute() and '..' not in entry.parts, name
            if entry.name in ('RECORD','INSTALLER','REQUESTED','direct_url.json','uv_cache.json'):
                continue
            content = Path(dist.locate_file(entry)).read_bytes()
            out.writestr(name, content)
            digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b'=').decode()
            rows.append([name, 'sha256='+digest, str(len(content))])
        record = 'transformer_engine_torch-2.20.2.dist-info/RECORD'
        rows.append([record,'',''])
        stream = io.StringIO(); csv.writer(stream).writerows(rows)
        out.writestr(record, stream.getvalue())
    import torch
    wheel.with_suffix('.json').write_text(json.dumps(dict(
        wheel=wheel.name, sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
        torch=torch.__version__,cuda=torch.version.cuda,cxx11_abi=torch._C._GLIBCXX_USE_CXX11_ABI,
        source='Unmodified PyPI transformer-engine-torch 2.20.2 sdist, repacked installed extension; retained to avoid repeated C++ builds',
        files=[r[0] for r in rows]),indent=2)+'\n')


if __name__ == '__main__': main()
