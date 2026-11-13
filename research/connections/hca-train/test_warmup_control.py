"""Stdlib checks of the one-run budget and fail-closed warmup gate."""
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from nanochat.hca_schedule import ratio_loss_weight
from warmup_preflight import PLAN,WARMUP,STEPS,TOKENS,validate_source,validate_trajectory
from warmup_run import launcher
import optimized_pair


def source():
    return dict(plan=PLAN,formal_training=True,attention_mode='dynamic_hca',steps=STEPS,training_tokens=TOKENS,
        seed=42,microbatch=16,batch_tokens=32768,initialization='seed42_from_zero_no_validation_weights',
        hca_backend='triton',hca=dict(target_ratio=4,window=256,ratio_loss_weight=1,temperature=1,
            noise=False,dynamic_gradient='confidence',router_init='uncompressed',ratio_warmup_steps=WARMUP))


class WarmupControl(unittest.TestCase):
    def test_update_index_and_legacy_default(self):
        for step, expected in ((0,0.),(1,.0005),(1000,.5),(1999,.9995),(2000,1.),(100000,1.)):
            self.assertEqual(ratio_loss_weight(1.,2000,step),expected)
        self.assertEqual(ratio_loss_weight(1.,0,None),1.)
        for bad in (None,-1,True,1.5):
            with self.assertRaises(ValueError): ratio_loss_weight(1.,2000,bad)

    def test_source_rejects_budget_or_recipe_changes(self):
        s=source();validate_source(s)
        for key,val in (('steps',STEPS+1),('training_tokens',TOKENS+32768),('attention_mode','fixed_hca')):
            bad=copy.deepcopy(s);bad[key]=val
            with self.assertRaises(AssertionError):validate_source(bad)
        for key,val in (('router_init','default'),('ratio_warmup_steps',0),('noise',True)):
            bad=copy.deepcopy(s);bad['hca'][key]=val
            with self.assertRaises(AssertionError):validate_source(bad)

    def test_gate_detects_warmup_restart_and_wrong_initial_rate(self):
        s=source();phases=[]
        for start,stop in ((0,1000),(1000,3000)):
            phases.append(dict(complete=True,formal_training=False,status='finite_limit',source=s,
                steps=stop,resumed_step=start,args=dict(seed=42,block_size=4,gradient='confidence',
                    resume=None if start==0 else '../diagnostic-state/latest.pt'),
                config=dict(hca_router_init='uncompressed',hca_ratio_warmup_steps=WARMUP,
                            hca_ratio_loss_weight=1,hca_noise=False),
                initial_validation=dict(step=0,hard_keep_rate=1),
                validation=[dict(step=stop,tokens=131072,ce=4.,hard_keep_rate=.25)]))
        rows=[dict(step=i+1,loss=4.,ce=4.,objective=4.,hard_keep_rate=1. if i==0 else .25,
                   seconds=.1,ratio_loss_weight=min(i/2000,1.)) for i in range(3000)]
        validate_trajectory(*phases,rows,s)
        rows[1000]['ratio_loss_weight']=0
        with self.assertRaises(AssertionError):validate_trajectory(*phases,rows,s)
        rows[1000]['ratio_loss_weight']=.5;rows[0]['hard_keep_rate']=.5
        with self.assertRaises(AssertionError):validate_trajectory(*phases,rows,s)

    def test_launcher_gate_precedes_fresh_formal_training(self):
        s=launcher()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'launch.sh';p.write_text(s);subprocess.run(['bash','-n',str(p)],check=True)
        self.assertIn('--hca-router-init uncompressed --hca-ratio-warmup-steps 2000',s)
        self.assertIn('--steps 305176 --schedule-steps 305176',s)
        self.assertIn('--wandb-group '+PLAN,s)
        self.assertIn('--resume auto --segment-seconds 1800',s)
        self.assertLess(s.index('warmup_preflight.py'),s.index('export WANDB_MODE=online'))
        self.assertNotIn('--resume ../diagnostic',s)

    def test_single_run_root_reuses_existing_state_without_duplicate_submission(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);run=root/'dynamic_hca';run.mkdir()
            plan=dict(commit='test',runs={'dynamic_hca':dict(root=str(run),run_id='one')})
            state=dict(status='complete',job='1.0',formal_started=True)
            (root/'plan.json').write_text(json.dumps(plan));(run/'state.json').write_text(json.dumps(state))
            with patch.object(optimized_pair,'command',side_effect=AssertionError('Unexpected scheduler mutation/query')):
                self.assertEqual(optimized_pair.enqueue('dynamic_hca',plan_root=root),state)
                self.assertFalse(optimized_pair.tick(plan_root=root,modes=('dynamic_hca',)))
            self.assertEqual(json.loads((root/'plan.json').read_text())['status'],'complete')


if __name__=='__main__':unittest.main()
