"""Finite-evidence launch condition checks; standard-library only, no GPU/job."""
import copy
import unittest
from workflow import validate_gate


def evidence():
    source = dict(commit='fixture',formal_training=False)
    profiles = {}
    for name,backend in (('baseline','auto'),('state_tokens','triton'),('state_tokens_torch','auto')):
        profiles[name] = dict(complete=True,causality_passed=True,causal_prefix_max_abs=0,
            optimizer_updates=60,losses=[5.]*60,training_tokens=60*32768,batch_tokens=32768,
            sequence_length=2048,seed=42,microbatch=16,source=source,gpu_uuid='fixture',
            input_sha256='fixture',logging_backend='local',mean_seconds=.2,peak_reserved_bytes=10,
            device_total_bytes=100,model=dict(variant='baseline' if name=='baseline' else 'state_tokens',
                                            state_write_residual=True,state_attention_backend=backend))
    report=dict(complete=True,formal_training=False,source=source,microbatch=16,profiles=profiles,
                checks=dict(tests=26,failures=0,errors=0,skipped=0),
                restart=dict(complete=True,steps=50,training_tokens=50*32768,source=source,
                             variant='state_tokens',initial_dev_ce=10.,final_dev_ce=6.,best_dev_ce=6.,peak_reserved_bytes=10))
    return report,dict(source=source)


class LaunchCondition(unittest.TestCase):
    def test_complete_finite_evidence(self):
        report,state=evidence()
        self.assertIs(validate_gate(report,state),report)

    def test_nonfinite_loss_on_either_backend(self):
        for name in ('state_tokens','state_tokens_torch'):
            for value in (float('nan'),float('inf'),-float('inf')):
                with self.subTest(backend=name,value=value):
                    report,state=evidence();report['profiles'][name]['losses'][-1]=value
                    with self.assertRaises(AssertionError):validate_gate(report,state)

    def test_incomplete_or_wrong_model(self):
        for path,value in ((('checks','failures'),1),(('restart','steps'),25),
                           (('restart','final_dev_ce'),float('nan')),
                           (('profiles','state_tokens','complete'),False),
                           (('profiles','state_tokens','model','state_write_residual'),False),
                           (('profiles','state_tokens','model','state_attention_backend'),'auto')):
            with self.subTest(path=path):
                report,state=evidence();target=report
                for k in path[:-1]:target=target[k]
                target[path[-1]]=value
                with self.assertRaises(AssertionError):validate_gate(report,state)

    def test_source_mismatch(self):
        report,state=evidence();state=copy.deepcopy(state);state['source']['commit']='different'
        with self.assertRaises(AssertionError):validate_gate(report,state)


if __name__=='__main__':unittest.main()
