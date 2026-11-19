import math

import pytest
import torch

from nanochat.connection_model import ConnectionConfig
from nanochat.geodesic_residual import geodesic_split_write, sphere_exp_transport, sphere_log_target
from nanochat.momentum import normalize_residual, tangent_project
from test_axis_write import test_active_write_cache_and_compiled_training
from test_momentum import gpu, test_exact_optimizer_checkpoint_next_update


@pytest.fixture
def setting():
    return ConnectionConfig(depth=2, width=128, heads=1, sequence_length=32, vocab_size=256,
                            streams=1, residual_dynamics='ngpt_lookahead', lookahead_mu_init=.3,
                            ngpt_velocity_mode='lookahead_geodesic_split', ngpt_force_rms_floor=1e-6)


@pytest.mark.parametrize('perp,parallel', [(.25,.25),(.2,.7)])
def test_split_arc_endpoint_and_arrival_velocity(perp,parallel):
    ex,ey,ez,_ = torch.eye(4,device='cuda',dtype=torch.float64)
    h, v = 2*ex, .6*ey
    angle = math.pi/3
    target = math.cos(angle)*ex + math.sin(angle)*(.6*ey + .8*ez)
    x, memory = geodesic_split_write(h,v,3.7*target,perp,parallel,.5,1e-6)
    s = 2*angle*(parallel*.6*ey + perp*.8*ez)
    distance = s.norm()
    e = s/distance
    turn = distance/2
    arrival = -turn.sin()*ex + turn.cos()*e
    torch.testing.assert_close(x,2*(turn.cos()*ex + turn.sin()*e))
    carried = v + v.dot(e)*(arrival-e)
    torch.testing.assert_close(memory,.5*(carried+distance*arrival))
    if perp == parallel:
        torch.testing.assert_close(turn,turn.new_tensor(math.radians(15)))


def test_transport_follows_the_selected_arc_even_at_a_half_or_full_turn():
    ex,ey,ez,_ = torch.eye(4,device='cuda',dtype=torch.float64)
    for theta in (0.,1e-10,math.pi,2*math.pi):
        step = 2*theta*ey
        x,v = sphere_exp_transport(2*ex,step,.5*(ey+ez+step))
        expected_tangent = -math.sin(theta)*ex + math.cos(theta)*ey
        torch.testing.assert_close(x,2*(math.cos(theta)*ex+math.sin(theta)*ey))
        torch.testing.assert_close(v,.5*((1+2*theta)*expected_tangent + ez))


def test_singular_inputs_and_log_exp_roundtrip():
    ex,ey,_,_ = torch.eye(4,device='cuda')
    h = 2*ex
    for f in (h, -h, -h+1e-4*ey, 0*h, 1e-11*ey):
        force = f.detach().requires_grad_()
        v = torch.zeros_like(h,requires_grad=True)
        log = sphere_log_target(h,force,1e-6)
        x,memory = geodesic_split_write(h,v,force,1.,1.,.5,1e-6)
        if f.norm() >= 1e-5:
            torch.testing.assert_close(x,normalize_residual(f),rtol=1e-6,atol=2e-6)
        if torch.equal(f,-h):
            torch.testing.assert_close(log.norm(),log.new_tensor(2*math.pi))
        if f.norm() == 0:
            assert torch.equal(x,h) and torch.count_nonzero(memory)==0
        assert torch.isfinite(log).all() and abs((x*memory).mean().item()) < 1e-5
        grads = torch.autograd.grad(x.sum()+memory.sum(),(v,force))
        assert all(torch.isfinite(g).all() for g in grads)


def test_arc_backward_for_inputs_and_both_step_sizes():
    def recurrence(raw_h, raw_v, f, a, b):
        h = normalize_residual(raw_h)
        return geodesic_split_write(h,tangent_project(h,raw_v),f,a,b,.5,1e-6)

    vectors = tuple(torch.randn(2,8,device='cuda',dtype=torch.float64,requires_grad=True) for _ in range(3))
    steps = tuple(vectors[0].new_tensor(t,requires_grad=True) for t in (.27,.56))
    assert torch.autograd.gradcheck(recurrence,vectors+steps)
