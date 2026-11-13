"""GPU-only checks that rejected updates retain useful, finite metadata."""
from types import SimpleNamespace
import json
import pytest
import torch
from nanochat.packed_training import train_step, NonfiniteTrainingError


@pytest.mark.parametrize('failure', ['gradient', 'loss', None])
def test_nonfinite_update_is_rejected_with_location(failure):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones((), device='cuda'))
            self.config = SimpleNamespace(sequence_length=2, attention_mode='full')

        def forward(self, x, y):
            return self.weight * (float('nan') if failure == 'loss' else 1.)

    model = Model()
    if failure == 'gradient':
        model.weight.register_hook(lambda grad: grad * float('inf'))
    class Data:
        def batch(self, split, start, n):
            assert start == 7
            return (torch.zeros(n, 2, device='cuda', dtype=torch.long),) * 2
    opt = torch.optim.SGD(model.parameters(), lr=.1)
    if failure:
        with pytest.raises(NonfiniteTrainingError) as caught:
            train_step(model, opt, Data(), 7, microbatch=1, batch_tokens=2)
        report = caught.value.report
        assert report['failed_update'] == 8 and report['next_data_row'] == 7
        assert report['loss_finite'] == (failure == 'gradient')
        assert report['nonfinite_gradient_parameters'] == ['weight']
        assert report['optimizer_updated'] is False
        assert model.weight.item() == 1.
        json.dumps(report, allow_nan=False)
    else:
        assert train_step(model, opt, Data(), 7, microbatch=1, batch_tokens=2) == 1.
        assert model.weight.item() == pytest.approx(.9)
