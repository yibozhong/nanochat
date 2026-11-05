"""Once-per-hardware-path checks before formal steps; no W&B or saved weights."""
import json,math,os,subprocess,sys
from pathlib import Path
import xml.etree.ElementTree as ET

import torch

from nanochat.connection_experiment import configure_attention,match_vanilla_initialization
from nanochat.connection_model import ConnectionConfig,ConnectionModel,optimizer_for,schedule
from nanochat.packed_training import PackedData,seed_all,json_write,train_step


def main():
    source=json.loads(Path('source.json').read_text())
    backend=configure_attention('hardware_auto')
    capability=list(torch.cuda.get_device_capability())
    root=Path('results/hardware-preflight');root.mkdir(parents=True,exist_ok=True)
    marker=root/(backend+'.json')
    identity=dict(commit=source['commit'],backend=backend,capability=capability,torch=torch.__version__)
    if marker.exists():
        old=json.loads(marker.read_text())
        if old.get('complete') and old.get('identity')==identity:
            print('Reusing completed hardware preflight: '+backend,flush=True);return
    env=dict(os.environ,WANDB_MODE='disabled')
    for key in ('WANDB_API_KEY','NETRC','WANDB_PROJECT','WANDB_ENTITY'):
        env.pop(key,None)
    xml=root/(backend+'-tests.xml')
    args=[sys.executable,'-m','pytest','-v','tests/test_stream_mlp.py','tests/test_stream_mlp_backend.py','--junitxml='+str(xml)]
    subprocess.run(args,env=env,check=True)
    suites=ET.parse(xml).getroot().findall('testsuite')
    counts={k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    assert counts==dict(tests=12,failures=0,errors=0,skipped=0),counts
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;seed_all(42)
    model=ConnectionModel(ConnectionConfig(variant='mhc_stream_mlp')).prepare(compile_blocks=True)
    match_vanilla_initialization(model,42);opt,wd=optimizer_for(model)
    data=PackedData('../data');torch.cuda.reset_peak_memory_stats();losses=[]
    for step in range(4):
        schedule(opt,step,305176,wd)
        losses.append(train_step(model,opt,data,step,16,32768))
    assert all(math.isfinite(x) for x in losses)
    assert all(torch.isfinite(p).all() for p in model.parameters())
    peak=torch.cuda.max_memory_reserved();assert peak<int(.9*22000*1024**2),peak
    result=dict(complete=True,identity=identity,checks=counts,gpu=torch.cuda.get_device_name(),
                preflight_losses=losses,peak_reserved_bytes=peak,formal_training=False)
    json_write(marker,result)
    print('Hardware preflight passed: '+json.dumps(result),flush=True)


if __name__=='__main__':main()
