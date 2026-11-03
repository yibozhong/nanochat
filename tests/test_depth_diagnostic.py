"""Small independent oracles; real-checkpoint forward/skip checks run in the diagnostic."""
import torch
from scripts.connection_depth_diagnostic import vectors, cosine, measurements, sublayer, sequence_statistics, summarized


def test_full_stream_norm_does_not_collapse_opposite_streams():
    x = torch.tensor([[[[1., -1.], [2., -2.]]]])
    assert vectors(x).shape == (1, 1, 4)
    torch.testing.assert_close(vectors(x).norm(dim=-1), torch.tensor([[10.**.5]]))
    assert x.mean(-1).norm() == 0


def test_known_ratios_cosines_and_decomposition():
    h = torch.tensor([[[3.,4.],[0.,2.]]])
    c = h.clone(); r = -h/2; after = h+c+r
    row, error = measurements(h, after, c, r)
    assert error == 0
    torch.testing.assert_close(row['total_ratio'], torch.full((1,2),.5))
    torch.testing.assert_close(row['compute_ratio'], torch.ones(1,2))
    torch.testing.assert_close(cosine(c,-c), -torch.ones(1,2))
    assert torch.isnan(cosine(c,torch.zeros_like(c))).all()


def test_nonidentity_routing_is_preserved_when_branch_disabled():
    class Routing:
        def read(self, x):
            post = torch.tensor([.25,.75]).expand(*x.shape[:2],2)
            return x.sum(-1), x, (post, None, None)
        def write(self, y, x, mixes):
            return x.flip(-1) + y.unsqueeze(-1)*torch.tensor([.25,.75])
    x = torch.tensor([[[[1.,3.],[2.,5.]]]])
    def forbidden(h):
        raise AssertionError('Disabled branch was executed')
    result, info = sublayer(x,Routing(),forbidden,True,True)
    torch.testing.assert_close(result,x.flip(-1),atol=0,rtol=0)
    assert not torch.equal(result,x)
    assert info[0].count_nonzero() == 0
    torch.testing.assert_close(info[1],result-x,atol=0,rtol=0)
    result, info = sublayer(x,Routing(),lambda h:h*2,False,True)
    torch.testing.assert_close(result-x,info[0]+info[1],atol=0,rtol=0)


def test_plain_skip_is_identity():
    x = torch.tensor([[[1.,3.],[2.,5.]]])
    out, info = sublayer(x,None,lambda h:1/0,True,True)
    assert torch.equal(out,x)
    assert info[0].count_nonzero()==info[1].count_nonzero()==0


def test_undefined_cosine_is_excluded_and_counted_not_zero_imputed():
    rows=[dict(update_residual_cosine=torch.tensor([[float('nan'),1.,-1.],[float('nan')]*3]),
               total_ratio=torch.zeros(2,3))]
    result=sequence_statistics(rows)
    assert result['update_residual_cosine_valid_tokens'][:,0].tolist()==[2,0]
    assert result['update_residual_cosine'][0,0]==0
    stats=summarized(result['update_residual_cosine'][:,0])
    assert stats['valid_sequences']==1 and stats['mean']==0
    assert summarized(result['update_residual_cosine'][1:,0])['mean'] is None
