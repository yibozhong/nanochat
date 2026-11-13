import copy,unittest
from run_policy import MODEL,PLAN,STEPS,TOKENS,validate_source,validate_gate,finite_formal_row


class PolicyTests(unittest.TestCase):
    def source(self):
        return dict(plan=PLAN,formal_training=True,steps=STEPS,training_tokens=TOKENS,
            batch_tokens=32768,microbatch=16,seed=42,model=MODEL,attention_backend='fa2',
            initialization='seed42_from_zero_no_diagnostic_weights',checkpoint_seconds=1800,
            evaluate_test=True,data_uri='osdf:///chtc/staging/y/yzhong239/nanochat-2k-10b-data-7685535.tar',
            dataset_revision='87f09149ef4734204d70ed1d046ddc9ca3f2b8f9')

    def test_recipe(self):
        s=self.source();validate_source(s)
        for key,value in [('steps',STEPS+1),('microbatch',8),('seed',43),('checkpoint_seconds',3600)]:
            c=copy.deepcopy(s);c[key]=value
            with self.assertRaises(AssertionError):validate_source(c)
        c=copy.deepcopy(s);c['model']['vocab_gate']=False
        with self.assertRaises(AssertionError):validate_source(c)

    def test_failed_validation_never_approves(self):
        s=self.source()
        r=dict(complete=True,source=s,tests_failed=0,tests_passed=48,finite_training_steps=200,
            cli_resume_passed=True,causal_cache_passed=True,optimizer_rng_resume_passed=True,
            gate_gradient_nonzero=True,trained_gate_nonzero=True,baseline_step_ms=60,gate_step_ms=90)
        validate_gate(r,s)
        for key,value in [('tests_failed',1),('tests_passed',47),('finite_training_steps',199),
                          ('cli_resume_passed',False),('causal_cache_passed',False),('gate_step_ms',float('nan'))]:
            c=copy.deepcopy(r);c[key]=value
            with self.assertRaises(AssertionError):validate_gate(c,s)

    def test_formal_rows(self):
        self.assertTrue(finite_formal_row(dict(step=1,tokens=32768,loss=10.)))
        self.assertFalse(finite_formal_row(dict(step=0,tokens=0,loss=10.)))
        self.assertFalse(finite_formal_row(dict(step=1,tokens=32768,loss=float('nan'))))


if __name__=='__main__':unittest.main()
