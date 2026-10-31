"""Logging policy checks use no GPU, network, credentials, or actual W&B runs."""
import builtins
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from nanochat.experiment_logging import diagnostic_run, training_run


class LoggingPolicyTest(unittest.TestCase):
    def test_diagnostics_never_import_wandb(self):
        original = builtins.__import__
        def guard(name, *args, **kwargs):
            if name == 'wandb' or name.startswith('wandb.'):
                raise AssertionError('Diagnostic imported W&B')
            return original(name, *args, **kwargs)
        with tempfile.TemporaryDirectory() as d, patch('builtins.__import__', side_effect=guard):
            for make in (diagnostic_run, training_run):
                r = make(local_dir=d, config={'seed': 42})
                r.log({'loss': 1.}); r.summary.update(complete=True); r.finish()
                self.assertEqual(json.loads((r.directory/'summary.json').read_text())['logging_backend'], 'local')
            self.assertFalse(list(Path(d).rglob('*.wandb')))

    def test_formal_training_opt_in_and_offline_fallback(self):
        calls=[]
        def init(**kwargs):
            calls.append(kwargs)
            if len(calls)==1:raise ConnectionError('simulated offline')
            return 'mock-run'
        wb=types.SimpleNamespace(init=init,Settings=lambda **k:k,errors=types.SimpleNamespace(CommError=ConnectionError))
        with patch.dict(sys.modules,wandb=wb):
            self.assertEqual(training_run(formal_training=True,id='mock-only',project='wrong'), 'mock-run')
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1]['mode'], 'offline')
        self.assertTrue(all(c['project']=='nanochat' and c['entity']=='yibozhong657-none' and c['id']=='mock-only' for c in calls))


if __name__=='__main__':unittest.main()
