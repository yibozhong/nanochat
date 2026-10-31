"""Only explicitly designated formal training may create a W&B run."""
import json
from pathlib import Path
import time
import uuid


class LocalRun:
    """Plain JSON diagnostics with the small recorder interface used by scripts."""
    def __init__(self, **kwargs):
        self.id = kwargs.get('id') or uuid.uuid4().hex[:8]
        self.directory = Path(kwargs.get('local_dir', 'results/diagnostics')) / self.id
        self.directory.mkdir(parents=True, exist_ok=True)
        self.config = dict(kwargs.get('config') or {})
        self.summary = {}
        self.metadata = {k: kwargs.get(k) for k in ('name', 'group', 'job_type')}
        self._write('config.json', self.config)

    def _write(self, name, value):
        path = self.directory / name
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(value, indent=2) + '\n')
        tmp.replace(path)

    def define_metric(self, *args, **kwargs):
        pass

    def log(self, values, **kwargs):
        with (self.directory / 'events.jsonl').open('a') as f:
            f.write(json.dumps(dict(time=time.time(), values=values, **kwargs)) + '\n')

    def save(self, *args, **kwargs):
        # Reports are already written to local paths by callers. No uploads.
        pass

    def finish(self, exit_code=0, **kwargs):
        self._write('config.json', self.config)
        self._write('summary.json', dict(id=self.id, logging_backend='local',
            exit_code=exit_code, metadata=self.metadata, summary=self.summary))


def diagnostic_run(**kwargs):
    """Never imports the SDK, contacts W&B, or creates offline W&B files."""
    return LocalRun(**kwargs)


def training_run(*, formal_training=False, **kwargs):
    if not formal_training:
        return diagnostic_run(**kwargs)
    import wandb
    kwargs.pop('local_dir', None)
    kwargs.update(project='nanochat', entity='yibozhong657-none')
    try:
        return wandb.init(**kwargs, settings=wandb.Settings(init_timeout=60))
    except wandb.errors.CommError:
        # Formal training alone may retain offline records for later sync.
        return wandb.init(**kwargs, mode='offline')
