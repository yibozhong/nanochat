"""Single-GPU FineWeb-Edu training: vanilla, mHC, or no-manifold ablation.

Example: python -m scripts.connection_train --variant mhc --steps 50 --data-dir data
Diagnostics require an explicit step count. Formal runs default to the 1B protocol.
"""
import argparse
from dataclasses import asdict
from functools import partial
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid

import torch
from nanochat.experiment_logging import BATCH_TOKENS, PROJECT, STEPS, training_run
from nanochat.procedural import DYCK, TASKS, ProceduralData

from nanochat.connection_model import ConnectionConfig, build_connection_model, optimizer_for, schedule, CONNECTION_VARIANTS
from nanochat.packed_training import PackedData, seed_all, evaluate, evaluate_procedural, train_step, snapshot, atomic_save, restore_rng, json_write
from nanochat.connection_experiment import configure_attention, match_vanilla_initialization


def check_resume_protocol(state, source, protocol, *, latest):
    if state['source'] == source:
        assert state['protocol'] == protocol
        return False
    parent = source.get('continuation')
    assert parent and state['source'] == parent['source']
    assert state['run_id'] == source['run_id'] == parent['source']['run_id']
    assert parent['step'] == parent['source']['steps'] < protocol['steps'] == protocol['schedule_steps']
    assert state['protocol'] == dict(protocol, steps=parent['step'])
    assert 0 < state['step'] <= parent['step']
    if latest:
        assert state['step'] == parent['step']
    return True


@torch.no_grad()
def velocity_usage_metrics(model, data, microbatch, normal_ce):
    ids, _ = data.batch('dev', 0, microbatch)
    _, v0 = model.initial_state(model.wte(ids), ids)
    v0_rms = v0.float().square().mean().sqrt().item()
    mu = model.blocks[0].attn_residual.mu.item()
    weight = model.velocity_embedding.weight
    table_rms = torch.linalg.vector_norm(weight, dtype=torch.float32).item() / math.sqrt(weight.numel())
    # Only replace the embedding lookup during this dev pass. Training weights
    # and optimizer state are never touched.
    with model.velocity_embedding.register_forward_hook(lambda module, args, output: torch.zeros_like(output)):
        zero_ce, _ = evaluate(model, data, microbatch=microbatch)
    return {'dev/zero_v0_ce': zero_ce, 'dev/zero_v0_delta_ce': zero_ce - normal_ce,
            'residual/initial_tangent_velocity_rms': v0_rms, 'residual/first_read_offset_rms': abs(mu)*v0_rms,
            'residual/velocity_embedding_rms': table_rms}


def record_zero_init_gradients(model, optimizer, path):
    """Capture the first real NTP backward before Muon changes the gradients."""
    blocks = model.native.transformer.h if model.config.architecture == 'native' else model.blocks
    projections = [branch.c_proj.weight for block in blocks for branch in (block.attn, block.mlp)]
    if any(weight.count_nonzero().item() for weight in projections):
        raise ValueError('Expected zero attention and FFN output projections in every block')

    captured = False

    def capture(opt, args, kwargs):
        nonlocal captured
        if captured:
            return
        norms = torch.stack([weight.grad.float().norm() for weight in projections])
        total = torch.stack([p.grad.float().norm() for p in model.parameters() if p.grad is not None]).norm()
        if not torch.isfinite(total) or not (norms > 0).all():
            raise FloatingPointError('Invalid first NTP gradient at zero initialization')
        report = dict(total_norm=total.item(), output_projection_norms=norms.tolist())
        json_write(path, report)
        print('FIRST_NTP_GRADIENT', report, flush=True)
        captured = True

    optimizer.register_step_pre_hook(capture)


def main():
    p = argparse.ArgumentParser(fromfile_prefix_chars='@')
    p.add_argument('--variant', choices=CONNECTION_VARIANTS, required=True)
    p.add_argument('--sequence-length', type=int, default=2048)
    p.add_argument('--attention-kernel', choices=('original','fla_triton'), default='original')
    p.add_argument('--architecture', choices=('vanilla', 'native'), default='vanilla')
    p.add_argument('--latent-feedback', action='store_true')
    p.add_argument('--residual-dynamics', choices=('vanilla','heavy_ball','heavy_ball_rms','ngpt_heavy_ball','ngpt_hbformer','ngpt','ngpt_lookahead','lookahead_only'), default='vanilla')
    p.add_argument('--matrix-optimizer', choices=('muon', 'arc'), default='muon')
    p.add_argument('--arc-direction', choices=('raw', 'muon'), default='raw', help='Direction used at the spherical write')
    p.add_argument('--arc-step-scale', choices=('fan_out', 'radius'), default='fan_out',
                   help='Multiply matrix LR by sqrt(fan_out) or by the fixed initial Frobenius norm of each matrix')
    p.add_argument('--arc-mu', type=float, default=1., help='Lookahead in units of the displacement EMA')
    p.add_argument('--arc-beta', type=float, default=.5, help='Transported displacement EMA decay')
    p.add_argument('--velocity-init', choices=('zero','token','token_position','projected_h0'), default=None,
                   help='projected_h0 uses a shared bias-free transform of the normalized initial hidden state in spherical lookahead.')
    p.add_argument('--velocity-embedding-std', type=float, default=.02,
                   help='Velocity table initialization std, or target pre-projection velocity RMS for projected_h0.')
    p.add_argument('--native-chord-write', choices=('tied','three_scalar','token_vnorm','token_bounded','token_h'))
    p.add_argument('--native-chord-control-lr', type=float,
                   help='Absolute peak AdamW LR for Chord controls, excluding the velocity embedding')
    p.add_argument('--chord-query-only-lookahead', action='store_true', help='Read attention Q at the Chord preview and K/V at the current state')
    p.add_argument('--chord-attnres', action='store_true', help='Mix Chord previews with completed block endpoints before each branch')
    p.add_argument('--chord-history-correction', action='store_true',
                   help='Keep the Chord preview and add a zero-initialized gated history-only read before pre-norm')
    p.add_argument('--chord-history-updates', action='store_true',
                   help='Store the end-minus-start displacement of each completed Chord block')
    p.add_argument('--chord-attnres-backend', choices=('direct','packed'), default='direct')
    p.add_argument('--match-velocity-table-with-ffn', action='store_true',
                   help='Give the zero-velocity Chord FFNs exactly the parameter budget of the removed token table')
    p.add_argument('--velocity-embedding-lr', type=float,
                   help='Absolute peak AdamW LR for the velocity token table; follows the normal schedule')
    p.add_argument('--eval-zero-v0', action='store_true', help='Measure reliance on initial token velocity on packed dev')
    p.add_argument('--momentum-beta-init', type=float, default=.5)
    p.add_argument('--momentum-gamma-init', type=float, default=None)
    p.add_argument('--momentum-eta-init', type=float, default=None,
                   help='HBFormer fixes eta=1. Other modes learn eta from their configured initial value.')
    p.add_argument('--ngpt-velocity-mode', choices=('ambient','transport_rms','transport_layernorm','lookahead_transport','lookahead_axis_write','lookahead_split_write','lookahead_split_learned_beta','lookahead_geodesic_split','lookahead_token_single','lookahead_minimal'), default=None,
                   help='New nGPT momentum defaults to transport_rms; old checkpoints require ambient')
    p.add_argument('--momentum-velocity-rms-floor', type=float, default=.1)
    p.add_argument('--chord-update-based', action='store_true',
                   help='Split the normalized force itself along and across velocity before adding to h')
    p.add_argument('--chord-soft-force-tau', type=float,
                   help='Normalize Chord force by sqrt(mean(F^2)+tau^2); zero uses the historical hard RMS floor')
    p.add_argument('--normalize-minimal-force', action='store_true',
                   help='Use the default Chord lookahead read and RMS-normalize F before the no-alpha addition')
    p.add_argument('--ngpt-force-rms-floor', type=float, default=.1)
    p.add_argument('--ngpt-weight-norm', action=argparse.BooleanOptionalAction, default=None,
                   help='Project backbone weights at initialization and after optimizer steps (HBFormer default)')
    p.add_argument('--compile-lookahead-branches', action='store_true',
                   help='Compile additive lookahead reads, backbone branches and residual writes separately')
    p.add_argument('--fuse-chord-read-norm', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--recompute-chord-force', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--chord-state-dtype', choices=('bfloat16', 'float32'),
                   help='Storage for spherical Chord h/v. Project-only defaults to BF16; resume inherits the checkpoint.')
    p.add_argument('--chord-kernel', choices=('current', 'terminal', 'boundary', 'both'), default='current',
                   help='Execution path for layer-beta Chord (terminal write, fused next read, or both)')
    p.add_argument('--lookahead-mu-init', type=float, default=0.)
    p.add_argument('--fixed-first-attention-mu', action='store_true', help='Fix the first Attention lookahead at one')
    p.add_argument('--ngpt-alpha-init', type=float, default=.05)
    p.add_argument('--signed-chord-steps', action='store_true', help='Allow both split-chord step sizes to cross zero independently')
    p.add_argument('--token-chord-gates', action='store_true', help='Modulate positive split steps by independent per-token cosine gates')
    p.add_argument('--chord-velocity-geometry', choices=('parallel_transport','project_only','ambient'),
                   default='parallel_transport', help='How Chord carries velocity to the next sublayer')
    p.add_argument('--chord-six-controls', choices=('off', 'static', 'small_dynamic', 'learned_dynamic'), default='off',
                   help='Use independent velocity write gains with static or token-corrected controls')
    p.add_argument('--chord-velocity-write', action='store_true', help='Add a learned scalar rho times the old velocity before Chord retraction')
    p.add_argument('--channel-chord-coefficients', action='store_true', help='Learn both Chord step sizes and velocity decay independently per channel')
    p.add_argument('--token-beta-gate', action='store_true', help='Gate velocity decay by per-token tangent alignment')
    p.add_argument('--residual-projection-init', choices=('zero','normal'), default=None)
    p.add_argument('--residual-projection-init-std', type=float, default=.02,
                   help='Normal output-projection std before division by sqrt(2*depth)')
    p.add_argument('--feedback-eval-passes', type=int, default=4)
    p.add_argument('--feedback-source', choices=('top','layers'), default='top')
    p.add_argument('--feedback-attention-backend', choices=('torch','triton'), default='triton')
    p.add_argument('--window-pattern', default='SSSL', help='Native backbone only; vanilla always uses full attention')
    p.add_argument('--formal-training', action='store_true', help='Explicitly enable W&B for a formal training run')
    p.add_argument('--wandb-group', help='Research question (lowercase directory slug)')
    p.add_argument('--run-name', help='Short descriptive configuration name')
    p.add_argument('--baseline-run', help='Baseline ID in nanochat-1b, or self for the reference configuration')
    p.add_argument('--connection-backend', choices=('nvidia','nvidia_precise'), default='nvidia')
    p.add_argument('--fused-connection-grad', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--compact-routing', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--attnres-block-size', type=int, default=2,
                   help='Moonshot paper S: Attention/MLP sublayers per residual block; even >=2, default2 = one Transformer layer')
    p.add_argument('--steps', type=int)
    p.add_argument('--procedural-task', choices=TASKS)
    p.add_argument('--procedural-tokens', type=int, default=0,
                   help='Warm-up model positions within the total budget, rounded to the nearest full batch (positive budgets use at least one)')
    p.add_argument('--procedural-input-length', type=int, help='Defaults to 64 symbols or one full context for Dyck')
    p.add_argument('--schedule-steps', type=int, default=STEPS)
    p.add_argument('--microbatch', type=int, default=16)
    p.add_argument('--streams', type=int, default=None)
    p.add_argument('--sinkhorn-iterations', type=int, default=20)
    p.add_argument('--activation-checkpoint', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--data-dir', default='data')
    p.add_argument('--check-zero-init-gradients', action='store_true', help='Measure the first diagnostic NTP backward')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--checkpoint-dir', default=None)
    p.add_argument('--attention-backend', choices=('auto','sdpa_cudnn','fa3','fa2','hardware_auto'), default='auto')
    p.add_argument('--matched-backbone-init', action='store_true')
    p.add_argument('--evaluate-test', action='store_true')
    p.add_argument('--run-id', default=None)
    p.add_argument('--resume', default=None)
    p.add_argument('--segment-seconds', type=float, default=1800,
                   help='Exit85 for durable checkpoint upload; worker-local saves are not durable')
    p.add_argument('--segment-steps', type=int, default=0,
                   help='Bounded restart validation; 0 uses only the time/signal limit')
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    if a.steps is None:
        if not a.formal_training:
            p.error('Diagnostics require an explicit --steps budget')
        a.steps = STEPS
    if a.procedural_tokens < 0 or (a.procedural_tokens > 0 and a.procedural_task is None):
        p.error('A positive --procedural-tokens budget requires --procedural-task')
    if not a.procedural_task and a.procedural_input_length is not None:
        p.error('--procedural-input-length requires --procedural-task')
    warm_steps = max(1, (a.procedural_tokens + BATCH_TOKENS//2) // BATCH_TOKENS) if a.procedural_tokens else 0
    if warm_steps >= a.schedule_steps or (a.formal_training and warm_steps >= a.steps):
        p.error('Reserve at least one FineWeb update after the procedural warm-up')
    if a.streams is None:
        a.streams = 1 if a.variant in ('skip_scale','attnres_scale') else 4
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    assert a.sequence_length>0 and 32768%a.sequence_length==0
    rows_per_step=32768//a.sequence_length
    assert a.microbatch>0 and rows_per_step%a.microbatch==0
    assert 0<a.steps<=a.schedule_steps
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    ckpt=Path(a.checkpoint_dir) if a.checkpoint_dir else out/'checkpoints'
    ckpt.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    actual_attention_backend = configure_attention(a.attention_backend)
    seed_all(a.seed)
    if a.resume=='auto':
        a.resume=str(ckpt/'latest.pt') if (ckpt/'latest.pt').exists() else None
    if a.resume:
        state=torch.load(a.resume,map_location='cpu',weights_only=False)
        if a.chord_state_dtype is None:
            a.chord_state_dtype = state['config'].get('chord_state_dtype', 'float32')
    if a.chord_soft_force_tau is None:
        a.chord_soft_force_tau = state['config'].get('chord_soft_force_tau', 0.) if a.resume else 0.
    if a.residual_projection_init is None:
        if a.resume:
            a.residual_projection_init = state['config'].get('residual_projection_init')
        elif a.formal_training and a.schedule_steps == STEPS:
            a.residual_projection_init = 'zero'
    c=ConnectionConfig(variant=a.variant,streams=a.streams,sinkhorn_iterations=a.sinkhorn_iterations,
                       sequence_length=a.sequence_length, compact_routing=a.compact_routing,
                       attnres_block_size=a.attnres_block_size,
                       attention_kernel=a.attention_kernel,
                       architecture=a.architecture, window_pattern=a.window_pattern, native_chord_write=a.native_chord_write,
                       native_chord_control_lr=a.native_chord_control_lr,
                       latent_feedback=a.latent_feedback, feedback_eval_passes=a.feedback_eval_passes,
                       residual_dynamics=a.residual_dynamics, velocity_init=a.velocity_init,
                       velocity_embedding_std=a.velocity_embedding_std,
                       match_velocity_table_with_ffn=a.match_velocity_table_with_ffn,
                       chord_query_only_lookahead=a.chord_query_only_lookahead,
                       chord_attnres=a.chord_attnres, chord_attnres_backend=a.chord_attnres_backend,
                       chord_history_correction=a.chord_history_correction, chord_history_updates=a.chord_history_updates,
                       momentum_beta_init=a.momentum_beta_init, momentum_gamma_init=a.momentum_gamma_init,
                       momentum_eta_init=a.momentum_eta_init, ngpt_force_rms_floor=a.ngpt_force_rms_floor,
                       chord_update_based=a.chord_update_based, chord_soft_force_tau=a.chord_soft_force_tau,
                       normalize_minimal_force=a.normalize_minimal_force, fixed_first_attention_mu=a.fixed_first_attention_mu,
                       lookahead_mu_init=a.lookahead_mu_init, ngpt_alpha_init=a.ngpt_alpha_init, ngpt_weight_norm=a.ngpt_weight_norm,
                       ngpt_velocity_mode=a.ngpt_velocity_mode, signed_chord_steps=a.signed_chord_steps, token_chord_gates=a.token_chord_gates,
                       chord_six_controls=a.chord_six_controls, chord_velocity_write=a.chord_velocity_write, channel_chord_coefficients=a.channel_chord_coefficients, token_beta_gate=a.token_beta_gate, chord_velocity_geometry=a.chord_velocity_geometry, chord_state_dtype=a.chord_state_dtype,
                       fuse_chord_read_norm=a.fuse_chord_read_norm, recompute_chord_force=a.recompute_chord_force,
                       momentum_velocity_rms_floor=a.momentum_velocity_rms_floor,
                       residual_projection_init=a.residual_projection_init,
                       residual_projection_init_std=a.residual_projection_init_std,
                       feedback_source=a.feedback_source, feedback_attention_backend=a.feedback_attention_backend,
                       activation_checkpoint=a.activation_checkpoint,
                       connection_backend=a.connection_backend, fused_connection_grad=a.fused_connection_grad)
    data=PackedData(a.data_dir)
    assert data.manifest['sequence_length']==c.sequence_length
    assert max(a.steps-warm_steps, 0)*rows_per_step<=len(data.arrays['train'])
    procedural = None
    if warm_steps:
        if data.manifest['tokenizer']['vocab_size'] != c.vocab_size:
            p.error('Model vocabulary must match the packed FineWeb tokenizer')
        length = a.procedural_input_length
        if length is None:
            length = c.sequence_length if a.procedural_task in DYCK else 64
        try:
            procedural = ProceduralData(a.procedural_task, a.seed, c.vocab_size, c.sequence_length, length)
        except ValueError as error:
            p.error(str(error))
    # User preference: check geometry/file sizes without scanning file contents.
    for split, info in data.manifest['files'].items():
        expected=info['rows']*(c.sequence_length+1)*2
        assert Path(a.data_dir,split+'.bin').stat().st_size==expected,split
    if Path('source.json').exists():
        source=json.loads(Path('source.json').read_text())
    elif a.resume:
        source = state['source'].copy()
    else:
        source=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    legacy_resume = bool(a.formal_training and a.resume and 'wandb_project' not in state['source'])
    if a.formal_training and not legacy_resume:
        from nanochat.research_protocol import validate_identity
        try:
            group, name, baseline = validate_identity(a.wandb_group or source.get('group'),
                a.run_name or source.get('run_name'), a.baseline_run or source.get('baseline_run'))
        except ValueError as error:
            p.error(str(error))
        if source.get('wandb_project', PROJECT) != PROJECT:
            p.error('New formal runs use ' + PROJECT)
        identity = dict(wandb_project=PROJECT, group=group, run_name=name, baseline_run=baseline)
        for key, value in dict(identity, steps=a.steps, schedule_steps=a.schedule_steps, seed=a.seed).items():
            if key in source and source[key] != value:
                p.error(f'Formal argument {key} differs from the submission manifest')
        source.update(identity)
    prepare_options = dict(compile_blocks=True)
    if a.compile_lookahead_branches:
        if c.residual_mode != 'lookahead_only':
            p.error('--compile-lookahead-branches requires --residual-dynamics lookahead_only')
        prepare_options['compile_lookahead_branches'] = True
    if a.chord_kernel != 'current':
        if c.ngpt_velocity_mode != 'lookahead_split_learned_beta' or c.signed_chord_steps or c.token_chord_gates or c.token_beta_gate:
            p.error('--chord-kernel requires positive split steps and layerwise learned beta')
        prepare_options['chord_kernel'] = a.chord_kernel
    model=build_connection_model(c).prepare(**prepare_options)
    if a.matched_backbone_init:
        match_vanilla_initialization(model,a.seed)
    if a.eval_zero_v0 and (c.residual_mode != 'ngpt_lookahead' or c.velocity_init != 'token'):
        p.error('--eval-zero-v0 requires token-only spherical lookahead')
    new_optimizer=partial(optimizer_for, model, matrix_optimizer=a.matrix_optimizer, arc_mu=a.arc_mu, arc_beta=a.arc_beta,
                         arc_direction=a.arc_direction, arc_step_scale=a.arc_step_scale,
                         velocity_embedding_lr=a.velocity_embedding_lr)
    opt,wd=new_optimizer()
    if a.check_zero_init_gradients:
        if a.formal_training or a.resume:
            p.error('--check-zero-init-gradients requires a fresh local diagnostic')
        record_zero_init_gradients(model, opt, out/'first-gradients.json')
    initial_groups=[{k:v for k,v in group.items() if k!='params'} for group in opt.param_groups]
    protocol=dict(steps=a.steps,schedule_steps=a.schedule_steps,batch_tokens=32768,
                  microbatch=a.microbatch,grad_accum=rows_per_step//a.microbatch,seed=a.seed,
                  attention_backend=a.attention_backend,matched_backbone_init=a.matched_backbone_init,
                  evaluate_test=a.evaluate_test)
    if procedural:
        protocol['procedural'] = dict(generator='jiang-packed-v1', task=procedural.task, input_length=length,
            requested_tokens=a.procedural_tokens, steps=warm_steps, model_tokens=warm_steps*BATCH_TOKENS,
            fineweb_schedule_steps=a.schedule_steps-warm_steps, vocabulary='native tokenizer IDs',
            objective='within-example next token' if procedural.task in DYCK else 'output-only next token',
            validation='fixed separate RNG stream, same generator and symbol rules',
            validation_rows=len(data.arrays['dev']),
            transfer='all parameters in place; fresh optimizer and phase-local schedule')
    if a.velocity_embedding_lr is not None:
        protocol['velocity_embedding_peak_lr'] = a.velocity_embedding_lr
    if a.eval_zero_v0:
        protocol['eval_zero_v0'] = True
    if a.compile_lookahead_branches:
        protocol['lookahead_compilation'] = 'separate_branches'
    if a.matrix_optimizer == 'arc':
        protocol['matrix_optimizer'] = dict(name='arc', mu=a.arc_mu, beta=a.arc_beta,
            sphere='initial_frobenius_per_matrix', step_length='scheduled_matrix_lr * sqrt(fan_out)')
        if a.arc_step_scale == 'radius':
            protocol['matrix_optimizer']['step_length'] = 'scheduled_matrix_lr * initial_frobenius_radius'
        if a.arc_direction != 'raw':
            protocol['matrix_optimizer']['direction'] = a.arc_direction
    if c.latent_feedback:
        protocol['feedback']=dict(pass_mix=[.15,.85],sampler='numpy.default_rng([seed,zero_based_step])',
            earlier_passes='no_grad_detached',eval_passes=c.feedback_eval_passes)
    warm_supervised = 0
    procedural_val_ce = None
    step=0;best=float('inf');training_seconds=0.;prior_elapsed=0.;initial_dev=None
    prior_peak = 0
    training_flops = 0
    run_id=a.run_id or uuid.uuid4().hex[:8]
    resuming_parent=False
    if a.formal_training and 'continuation' in source:
        assert a.resume, 'The continuation requires its latest checkpoint'
    if a.resume:
        assert asdict(ConnectionConfig.from_checkpoint(state['config']))==asdict(c)
        assert state['data_manifest']==data.manifest
        resuming_parent=check_resume_protocol(state,source,protocol,latest=True)
        if a.run_id:assert state['run_id']==a.run_id
        run_id=state['run_id'];step=state['step'];best=state['best_dev_ce']
        training_seconds=state['training_seconds'];prior_elapsed=state['elapsed_seconds']
        initial_dev=state.get('initial_dev_ce')
        prior_peak = state.get('peak_allocated_bytes', 0)
        training_flops = state.get('estimated_training_flops', 0)
        if procedural:
            warm_supervised = state['procedural_supervised_tokens']
            procedural_val_ce = state.get('procedural_val_ce')
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
        restore_rng(state['rng']);del state
        assert (ckpt/'best.pt').exists(),'Resume requires the paired dev-best checkpoint'
    start_step=step
    config=dict(logging_backend='wandb' if a.formal_training else 'local',model=asdict(c),protocol=protocol,data_manifest=data.manifest,source=source,
        checkpoint_transport=dict(upload_interval_seconds=a.segment_seconds,local_save_every_steps=500,
            method='exit85 plus scheduler transfer; local save alone is not durable'),
        gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
        gpu_capability=list(torch.cuda.get_device_capability()),
        actual_attention_backend=actual_attention_backend,
        explicit_portability_gate=os.environ.get('NANOCHAT_MHC_TEST_PORTABILITY')=='1',
        parameters=sum(p.numel() for p in model.parameters()),
        residual_mixing=(c.residual_mode if c.residual_mode != 'vanilla' else
                         'sinkhorn' if a.variant=='mhc' else
                         'unconstrained_raw' if a.variant=='mhc_no_sinkhorn' else
                         'local_channel_unconstrained_raw' if a.variant=='mhc_channel_no_sinkhorn' else
                         'static_input_channel_scaled_shared_raw' if a.variant=='scaled_hc' else
                         'static_read_mix_post_head_channel_scaled_shared_raw' if a.variant=='scaled_hc_all' else
                         'block_attention_residuals_static_value_channel_scale' if a.variant=='attnres_scale' else
                         'block_attention_residuals' if a.variant=='attnres' else
                         'static_per_channel_identity_skip_scale' if a.variant=='skip_scale' else
                         'additive'),
        applied_sinkhorn_iterations=c.sinkhorn_iterations if a.variant=='mhc' else 0,
        optimizer=('Arc with Muon direction + native AdamW' if a.arc_direction == 'muon' else
                   'Arc matrix updates + native AdamW' if a.matrix_optimizer == 'arc' else
                   'native MuonAdamW; connector AdamW at native unembedding LR, no decay'),
        optimizer_groups=initial_groups,
        loss='full-token full-vocabulary CCE; no gradient filtering',
        references='research/connections/references.json')
    if a.chord_kernel != 'current':
        config['chord_kernel'] = a.chord_kernel
    from nanochat.connection_cost import model_cost
    cost=model_cost(c,batch=rows_per_step)
    config['compute_estimate']=cost
    nominal_flops = cost['estimated_training_flops'] if cost else None
    if nominal_flops is None and c.architecture == 'native':
        nominal_flops = model.native.estimate_flops() * protocol['batch_tokens']
        config['flops_scope'] = 'native GPT matrix and attention estimate; excludes connector arithmetic'
    elif cost:
        config['flops_scope'] = cost['accounting']
    else:
        d = c.width
        matmuls = c.depth*(4*d*d + 2*d*c.ffn_width) + d*c.vocab_size
        nominal_flops = 6*protocol['batch_tokens']*(matmuls + c.depth*d*(c.sequence_length+1))
        config['flops_scope'] = 'vanilla backbone matrix and causal attention estimate; excludes connector arithmetic'
    config['estimated_training_flops_per_step'] = nominal_flops
    if a.formal_training and not legacy_resume:
        from nanochat.research_protocol import comparison_protocol
        backbone = model.native if c.architecture == 'native' else model
        token_embedding = backbone.transformer.wte if c.architecture == 'native' else backbone.wte
        first_block = backbone.transformer.h[0] if c.architecture == 'native' else backbone.blocks[0]
        config['training_dtype'] = str(token_embedding.weight.dtype)
        anchors = {'input':token_embedding.weight, 'output':backbone.lm_head.weight, 'matrix':first_block.attn.c_q.weight}
        config['backbone_optimizer_groups'] = {
            name: initial_groups[next(i for i, group in enumerate(opt.param_groups) if any(p is weight for p in group['params']))]
            for name, weight in anchors.items()}
        config['comparison_protocol'] = comparison_protocol(config)
        config['arguments'] = vars(a).copy()
    if c.match_velocity_table_with_ffn and 'parameter_match' in source:
        match = source['parameter_match']
        assert config['parameters'] == match['total_parameters']
        assert all(b.mlp.c_fc.out_features == b.mlp.c_proj.in_features == match['ffn_width'] for b in model.blocks)
        assert cost['additional_ffn_parameters'] == match['velocity_table_parameters'] and model.velocity_embedding is None
    profile_reference=source.get('profile_reference',{}).get(a.variant)
    if profile_reference:
        config['profile_reference']=profile_reference
        config['profile_matches_current_gpu']=str(torch.cuda.get_device_properties(0).uuid)==profile_reference.get('gpu_uuid')
    kwargs=dict(group=a.wandb_group,
                job_type='pretrain' if a.formal_training else 'diagnostic',
                name=f'{a.architecture}-{a.variant}'+('-ngpt' if c.residual_mode == 'ngpt' else
                    f'-{c.residual_mode}-v0-{c.velocity_init}' if c.residual_mode != 'vanilla' else
                    '-layer-feedback' if c.feedback_source=='layers' else '-feedback' if c.latent_feedback else '')+
                    (f'-{c.native_chord_write}-write' if c.native_chord_write else '')+
                    ('-velocity-ln' if c.ngpt_velocity_mode == 'transport_layernorm' else '')+
                    ('-axis-write' if c.ngpt_velocity_mode == 'lookahead_axis_write' else '')+
                    ('-split-write' if c.split_chord_write else '')+
                    ('-update-based' if c.chord_update_based else '')+
                    (f'-soft-tau{c.chord_soft_force_tau:g}' if c.chord_soft_force_tau else '')+
                    ('-norm-force-no-alpha' if c.normalize_minimal_force else
                     '-minimal-static-raw' if c.ngpt_velocity_mode == 'lookahead_minimal' else '')+
                    ('-token-h-single-step' if c.ngpt_velocity_mode == 'lookahead_token_single' else '')+
                    ('-learned-beta' if c.ngpt_velocity_mode == 'lookahead_split_learned_beta' else '')+
                    ('-signed-steps' if c.signed_chord_steps else '')+
                    ('-token-alpha-gates' if c.token_chord_gates else '')+
                    ('-zero-init-velocity-embed' if c.velocity_init=='token' and c.velocity_embedding_std==0 else '')+
                    ('-fixed-first-attn-mu1' if c.fixed_first_attention_mu else '')+
                    (f'-chordattnres-s{c.attnres_block_size}' if c.chord_attnres else '')+
                    ('-history-correction' if c.chord_history_correction else '')+
                    ('-block-chords' if c.chord_history_updates else '')+
                    ('-q-lookahead-kv-current' if c.chord_query_only_lookahead else '')+
                    (f'-six-{c.chord_six_controls}' if c.chord_six_controls != 'off' else '')+
                    ('-rho-v' if c.chord_velocity_write else '')+
                    ('-channel-alpha-beta' if c.channel_chord_coefficients else '')+
                    (f'-ffn{c.ffn_width}-matched' if c.match_velocity_table_with_ffn else '')+
                    (f'-vembed-lr{a.velocity_embedding_lr:g}' if a.velocity_embedding_lr is not None else '')+
                    ('-token-beta-gate' if c.token_beta_gate else '')+
                    (f'-velocity-{c.chord_velocity_geometry}' if c.chord_velocity_geometry != 'parallel_transport' else '')+
                    ('-fused-read-norm' if c.fuse_chord_read_norm else '')+
                    ('-remat-force' if c.recompute_chord_force else '')+
                    ('-bf16-state' if c.chord_state_dtype == 'bfloat16' else '')+
                    ('-geodesic-split' if c.ngpt_velocity_mode == 'lookahead_geodesic_split' else '')+
                    ('-no-weight-norm' if c.residual_mode=='ngpt_hbformer' and not c.ngpt_weight_norm else '')+
                    ('-arc-muon-direction' if a.arc_direction == 'muon' else '-arc' if a.matrix_optimizer == 'arc' else '')+
                    ('-radius-step' if a.arc_step_scale == 'radius' else '')+
                    ('-mu0' if a.matrix_optimizer == 'arc' and a.arc_mu == 0 else '')+
                    (f'-init-{c.residual_projection_init}' if c.residual_projection_init != 'zero' or c.ngpt_velocity_mode == 'lookahead_minimal' else '')+
                    f'-{c.sequence_length}ctx-{a.steps}steps',id=run_id,resume='allow',config=config)
    if a.formal_training and not legacy_resume:
        kwargs.update(name=source['run_name'], group=source['group'])
    run=training_run(legacy_resume=legacy_resume, formal_training=a.formal_training, local_dir=out/"diagnostics", **kwargs)
    if a.formal_training and resuming_parent:
        run.config.update({'source':source, 'protocol':protocol}, allow_val_change=True)
        run.name=kwargs['name']
        run.summary.update(dict(complete=False, checkpoint_step=step, source=source, test_evaluated=False,
                                final_dev_ce=None, final_test_ce=None, best_selected_test_ce=None))
    run.define_metric('tokens')
    if procedural:
        run.define_metric('procedural/*', step_metric='tokens')
    for metric in ('train/*','dev/*','perf/*','residual/*'):
        run.define_metric(metric,step_metric='tokens')
    if a.matrix_optimizer == 'arc':
        run.define_metric('optimizer/*', step_metric='tokens')
    json_write(out/'config.json',config)
    stop=False;begin=time.perf_counter()
    def interrupted(signum,frame):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGUSR1,interrupted)
    procedural_dev = None
    if procedural and step <= warm_steps:
        procedural_dev = procedural.batch(0, len(data.arrays['dev']), split='dev')

    def token_counts():
        if not procedural:
            return {}
        return dict(procedural_tokens=min(step, warm_steps)*BATCH_TOKENS,
                    procedural_supervised_tokens=warm_supervised,
                    fineweb_tokens=max(0, step-warm_steps)*BATCH_TOKENS)

    def save():
        atomic_save(snapshot(model,opt,step,config=asdict(c),protocol=protocol,source=source,
            run_id=run_id,data_manifest=data.manifest,best_dev_ce=best,
            initial_dev_ce=initial_dev,training_seconds=training_seconds,**token_counts(),
            **(dict(procedural_val_ce=procedural_val_ce) if procedural else {}),
            peak_allocated_bytes=max(prior_peak,torch.cuda.max_memory_allocated()),estimated_training_flops=training_flops,
            elapsed_seconds=prior_elapsed+time.perf_counter()-begin),
            ckpt/'latest.pt')
        json_write(out/'progress.json',dict(run_id=run_id,step=step,tokens=step*32768,
            best_dev_ce=best,training_seconds=training_seconds,source=source,**token_counts()))
    def save_best(dev):
        atomic_save(dict(model=model.state_dict(),config=asdict(c),protocol=protocol,source=source,
            run_id=run_id,step=step,dev_ce=dev,data_manifest=data.manifest),ckpt/'best.pt')
    def dev_metrics(dev):
        nonlocal procedural_val_ce
        metrics={'dev/ce':dev, **token_counts()}
        if procedural and step <= warm_steps:
            procedural_val_ce, supervised = evaluate_procedural(model, procedural_dev, a.microbatch)
            metrics.update({'procedural/val_ce': procedural_val_ce, 'procedural/val_supervised_tokens': supervised})
            print(f'PROCEDURAL_DEV step={step} val_ce={procedural_val_ce:.6f} targets={supervised}', flush=True)
        if a.matrix_optimizer == 'arc':
            metrics.update(opt.metrics())
        if c.residual_mode != 'vanilla':
            metrics.update(model.residual_metrics())
        if a.eval_zero_v0 and (step in (250, 1000, a.steps) or step % 5000 == 0):
            usage = velocity_usage_metrics(model, data, a.microbatch, dev)
            metrics.update(usage)
            print('VELOCITY_USAGE ' + json.dumps(dict(step=step, normal_ce=dev, **usage)), flush=True)
        if c.latent_feedback:
            metrics[f'dev/ce_p{c.feedback_eval_passes}']=dev
            if c.feedback_eval_passes != 1:
                metrics['dev/ce_p1']=evaluate(model,data,microbatch=a.microbatch,passes=1)[0]
        return metrics
    try:
        torch.cuda.reset_peak_memory_stats()
        dev,dev_tokens=evaluate(model,data,microbatch=a.microbatch)
        if initial_dev is None:initial_dev=dev
        if step==0:
            best=dev;save_best(dev);save()
        if resuming_parent:
            expected=source['continuation']['dev_ce']
            assert abs(dev-expected)<1e-3, (step,dev,expected)
            print(f'RESUMED_PARENT step={step} dev_ce={dev:.8f}',flush=True)
        else:
            run.log(dict(step=step,tokens=step*32768,**dev_metrics(dev)))
        durations=[]
        with (out/'steps.jsonl').open('a',buffering=1) as log:
            while step<a.steps:
                warming = step < warm_steps
                phase_step = step if warming else step-warm_steps
                horizon = warm_steps if warming else a.schedule_steps-warm_steps
                lr=schedule(opt,phase_step,horizon,wd)
                torch.cuda.synchronize();start=time.perf_counter()
                batch = procedural.batch(step*rows_per_step, rows_per_step) if warming else None
                loss=train_step(model,opt,data,phase_step,microbatch=a.microbatch,seed=a.seed,masked_batch=batch)
                if warming:
                    warm_supervised += int(batch[2].sum())
                torch.cuda.synchronize();duration=time.perf_counter()-start
                durations.append(duration);training_seconds+=duration;step+=1
                row=dict(step=step,tokens=step*32768,loss=loss,seconds=duration,lr_multiplier=lr)
                if procedural:
                    row.update(phase='procedural' if warming else 'fineweb', phase_step=phase_step+1, **token_counts())
                if c.latent_feedback:
                    row['passes']=model.training_passes(a.seed,phase_step)
                step_cost = cost
                step_flops = nominal_flops
                if c.latent_feedback and cost:
                    step_cost = model_cost(c,batch=rows_per_step,feedback_passes=row['passes'])
                    step_flops = step_cost['estimated_training_flops']
                if step_flops is not None:
                    training_flops += step_flops
                log.write(json.dumps(row)+'\n')
                if step<=5 or step%10==0:
                    print(json.dumps(row),flush=True)
                    loss_key = 'procedural/ce' if warming else 'train/ce'
                    perf={loss_key:loss, **token_counts(),'perf/step_seconds':duration,'perf/tokens_per_second':32768/duration,
                          'perf/peak_allocated_bytes':max(prior_peak,torch.cuda.max_memory_allocated()),
                          'perf/training_seconds':training_seconds,
                          'perf/mean_tokens_per_second':row['tokens']/training_seconds,
                          'perf/peak_reserved_bytes':torch.cuda.max_memory_reserved()}
                    if cost:
                        perf['perf/estimated_forward_flops_per_step'] = step_cost['estimated_forward_flops']
                    if step_flops is not None:
                        perf.update({'perf/estimated_training_flops_per_step':step_flops,
                                     'perf/estimated_training_flops':training_flops})
                    if c.latent_feedback:
                        perf['train/passes']=row['passes']
                    if profile_reference and 'additional_peak_allocated_bytes' in profile_reference:
                        perf['perf/profile_additional_peak_allocated_bytes']=profile_reference['additional_peak_allocated_bytes']
                        perf['perf/profile_additional_peak_reserved_bytes']=profile_reference['additional_peak_reserved_bytes']
                    run.log(dict(step=step,tokens=row['tokens'],**perf))
                transfer = procedural is not None and step == warm_steps
                if transfer:
                    # The diagnostic verifies every tensor at this boundary on the real model.
                    before = None if a.formal_training else {
                        name: value.detach().clone() for name, value in model.state_dict().items()}
                    opt.zero_grad(set_to_none=True)
                    del opt
                    opt, wd = new_optimizer()
                    if before is not None:
                        after = model.state_dict()
                        assert before.keys() == after.keys() and not opt.state
                        for name, value in before.items():
                            assert torch.equal(value, after[name]), f'Weight changed at transfer: {name}'
                        json_write(out/'transfer-check.json', dict(step=step, identical_tensors=len(after),
                            optimizer_state_entries=len(opt.state), fineweb_next_row=0))
                        del before, after, value
                    print(f'PROCEDURAL_TRANSFER step={step} fineweb_next_row=0', flush=True)
                if transfer or step==50 or step%250==0 or step==a.steps:
                    dev,dev_tokens=evaluate(model,data,microbatch=a.microbatch)
                    assert math.isfinite(dev)
                    if dev<best or transfer:
                        best=dev;save_best(dev)
                    run.log(dict(step=step,tokens=step*32768,**dev_metrics(dev),**{'dev/tokens':dev_tokens}))
                if transfer or step==50 or step%500==0 or step==a.steps or stop:save()
                if step<a.steps and (stop or time.perf_counter()-begin>=a.segment_seconds
                                      or a.segment_steps and step-start_step>=a.segment_steps):
                    save();run.summary.update(dict(complete=False,checkpoint_step=step))
                    run.finish();return 85
        save()
        test_result={}
        if a.evaluate_test:
            final_test,test_tokens=evaluate(model,data,'test',microbatch=a.microbatch)
            selected=torch.load(ckpt/'best.pt',map_location='cpu',weights_only=False)
            assert selected['run_id']==run_id and selected['data_manifest']==data.manifest
            check_resume_protocol(selected,source,protocol,latest=False)
            model.load_state_dict(selected['model'])
            best_test,best_test_tokens=evaluate(model,data,'test',microbatch=a.microbatch)
            assert best_test_tokens==test_tokens
            test_result=dict(final_test_ce=final_test,best_selected_test_ce=best_test,
                best_dev_step=selected['step'],test_tokens=test_tokens)
        checkpoint_files={}
        for name in ('latest.pt','best.pt'):
            checkpoint_files[name]=dict(bytes=(ckpt/name).stat().st_size)
        result=dict(complete=True,variant=a.variant,run_id=run_id,steps=step,training_tokens=step*32768,
            initial_dev_ce=initial_dev,final_dev_ce=dev,best_dev_ce=best,dev_tokens=dev_tokens,
            training_seconds=training_seconds,elapsed_seconds=prior_elapsed+time.perf_counter()-begin,
            steady_step_seconds=sum(durations[10:])/len(durations[10:]) if len(durations)>10 else None,
            gpu=torch.cuda.get_device_name(),peak_allocated_bytes=max(prior_peak,torch.cuda.max_memory_allocated()),
            estimated_training_flops=training_flops if nominal_flops is not None else None,
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),source=source,
            test_evaluated=a.evaluate_test,checkpoint_files=checkpoint_files,**token_counts(),**test_result)
        if procedural:
            result['procedural_val_ce'] = procedural_val_ce
        json_write(out/'summary.json',result);run.summary.update(result);run.finish()
        print(json.dumps(result),flush=True);return 0
    except BaseException:
        run.finish(exit_code=1);raise


if __name__=='__main__':
    raise SystemExit(main())
