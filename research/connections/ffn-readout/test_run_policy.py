"""AP stdlib checks: authorization bounds, completion, and archive exit status."""
import copy
import json
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

from run_policy import PLAN,STEPS,TOKENS,BATCH_TOKENS,DATA_URI,DATA_REVISION,validate_source,validate_complete,finite_formal_row

HERE=Path(__file__).resolve().parent


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.e=json.loads((HERE/'results.json').read_text())
        self.s=dict(plan=PLAN,formal_training=True,steps=STEPS,training_tokens=TOKENS,
            batch_tokens=BATCH_TOKENS,microbatch=16,seed=42,attention_backend='fa2',
            initialization='seed42_from_zero_no_diagnostic_weights',data_uri=DATA_URI,
            dataset_revision=DATA_REVISION,checkpoint_seconds=1800,evaluate_test=True,
            model=self.e['workload'],run_id='one-run')

    def test_verified_recipe_accepted(self):
        validate_source(self.s,self.e)

    def test_budget_and_recipe_changes_rejected(self):
        for key,value in [('steps',STEPS+1),('microbatch',8),('seed',43),('attention_backend','fa3'),
                          ('formal_training',False),('dataset_revision','changed'),('evaluate_test',False)]:
            with self.subTest(key=key),self.assertRaises(AssertionError):
                s=copy.deepcopy(self.s);s[key]=value;validate_source(s,self.e)

    def test_model_changes_rejected(self):
        for key,value in [('ffn_readout_width',128),('ffn_readout_init','random'),('ffn_readout_backend','reference')]:
            with self.subTest(key=key),self.assertRaises(AssertionError):
                s=copy.deepcopy(self.s);s['model'][key]=value;validate_source(s,self.e)

    def test_formal_row_requires_finite_actual_step(self):
        row=dict(step=1,tokens=BATCH_TOKENS,loss=10.)
        self.assertTrue(finite_formal_row(row))
        for change in [dict(step=0),dict(tokens=1),dict(loss=float('nan')),dict(loss=float('inf'))]:
            self.assertFalse(finite_formal_row(dict(row,**change)))

    def test_completion_requires_budget_and_same_run(self):
        r=dict(complete=True,source=self.s,run_id='one-run',variant='baseline',steps=STEPS,
            training_tokens=TOKENS,test_evaluated=True,final_dev_ce=3.,final_test_ce=3.,
            best_selected_test_ce=3.,checkpoint_files={'latest.pt':{},'best.pt':{}})
        validate_complete(r,self.s)
        for change in [dict(steps=STEPS-1),dict(run_id='different'),dict(test_evaluated=False),dict(final_test_ce=float('nan'))]:
            with self.subTest(change=change),self.assertRaises(AssertionError):
                validate_complete(dict(r,**change),self.s)

    def test_archive_success_preserves_exit85(self):
        self.check_archive(False,85)

    def test_archive_failure_cannot_report_exit85(self):
        self.check_archive(True,74)

    def check_archive(self,fail,expected):
        fn=re.search(r'finish\(\) \{.*?\n\}',(HERE/'train.sh').read_text(),re.S).group()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'results').mkdir();(root/'checkpoints').mkdir()
            old=root/'checkpoints.tar';old.write_bytes(b'previous durable archive')
            script='set -euo pipefail\nTASK_ROOT="$PWD"\n'+fn+'\ntrap finish EXIT\n'
            if fail:script+='tar() { return 1; }\n'
            script+='exit 85\n'
            result=subprocess.run(['bash','-c',script],cwd=root,capture_output=True)
            self.assertEqual(result.returncode,expected,result.stderr)
            if fail:self.assertEqual(old.read_bytes(),b'previous durable archive')
            else:self.assertTrue((root/'results.tar').is_file())


if __name__=='__main__':unittest.main()
