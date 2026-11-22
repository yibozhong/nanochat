"""Submit one named nonzero-init residual run with durable restart."""
import argparse
import fcntl
import json
import math
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
import uuid

from policy import RANK, require_gpus

REPO=Path(__file__).resolve().parents[3]
HOME_ROOT=Path('/home/yzhong239')
ASSETS=HOME_ROOT/'nanochat-runtime-assets'
PRESETS={'mhc':'vanilla', 'attnres':'vanilla', 'arc':'vanilla', 'arc_muon':'vanilla', 'arc_muon_radius':'vanilla', 'vanilla':'vanilla', 'ngpt':'ngpt', 'ngpt_momentum':'ngpt_heavy_ball',
         'ngpt_momentum_transport_rms':'ngpt_heavy_ball', 'heavy_ball_rms':'heavy_ball_rms',
         'ngpt_hbformer':'ngpt_hbformer', 'ngpt_hbformer_no_wn':'ngpt_hbformer',
         'ngpt_hbformer_velocity_ln':'ngpt_hbformer', 'ngpt_lookahead':'ngpt_lookahead', 'lookahead_only':'lookahead_only',
         'ngpt_lookahead_axis_write':'ngpt_lookahead', 'ngpt_lookahead_split_write':'ngpt_lookahead',
         'ngpt_lookahead_split_learned_beta':'ngpt_lookahead',
         'ngpt_lookahead_geodesic_split':'ngpt_lookahead'}
STEPS=305176


def command(args,cwd=REPO):
    return subprocess.check_output(args,cwd=cwd,text=True,timeout=60).strip()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('preset',choices=PRESETS)
    parser.add_argument('--signed-chord-steps', action='store_true',
                        help='Compare two signed chord steps to the positive-step learned-beta run')
    parser.add_argument('--token-chord-gates', action='store_true',
                        help='Compare two token-dependent positive steps to the learned-beta control')
    parser.add_argument('--chord-velocity-geometry', choices=('project_only','ambient'),
                        help='Compare endpoint projection or unconstrained velocity to parallel transport')
    parser.add_argument('--fuse-chord-read-norm', action='store_true',
                        help='Benchmark separate, fused and force-recomputed Chord then train fused token-only from scratch')
    velocity = parser.add_mutually_exclusive_group()
    velocity.add_argument('--zero-velocity', action='store_true',
                          help='Start the standard scalar Chord from rest without a velocity table')
    velocity.add_argument('--token-only-velocity', action='store_true',
                        help='Remove the velocity position table from the project-only Chord control')
    parser.add_argument('--zero-init-velocity-embedding', action='store_true',
                        help='Initialize the trainable scalar Chord token velocity table to zero')
    parser.add_argument('--fixed-first-attention-mu', action='store_true',
                        help='Fix the first Attention lookahead at one in the standard Chord recipe')
    parser.add_argument('--chord-velocity-write', action='store_true',
                        help='Compare direct scalar rho*v writeback to the fused scalar Chord default')
    parser.add_argument('--channel-chord-coefficients', action='store_true',
                        help='Compare per-channel alpha and beta to the fused token-only BF16 Chord')
    parser.add_argument('--token-beta-gate', action='store_true',
                        help='Compare token-dependent velocity decay to the positive split learned-beta control')
    parser.add_argument('--hopper-only', action='store_true',
                        help='Require a full H100 or H200 for the entire run, including restarts')
    parser.add_argument('--no-arc-lookahead', action='store_true',
                        help='Run the radius-step Muon-direction control with gradients evaluated at the base weights')
    start=parser.add_mutually_exclusive_group()
    start.add_argument('--fresh-10b', action='store_true', help='Start an independent 10B lookahead run at step zero')
    start.add_argument('--extend-to-schedule', action='store_true', help='Continue the completed lookahead pilot to its original 10B schedule')
    args=parser.parse_args()
    preset=args.preset
    arc_optimizer = preset in ('arc', 'arc_muon', 'arc_muon_radius')
    if args.no_arc_lookahead and preset != 'arc_muon_radius':
        parser.error('--no-arc-lookahead requires arc_muon_radius')
    full_lookahead=args.fresh_10b or args.extend_to_schedule
    if full_lookahead and preset != 'ngpt_lookahead':
        parser.error('The lookahead start options require ngpt_lookahead')
    mode=PRESETS[preset]
    geodesic = preset == 'ngpt_lookahead_geodesic_split'
    learned_beta = preset == 'ngpt_lookahead_split_learned_beta'
    if args.signed_chord_steps and not learned_beta:
        parser.error('--signed-chord-steps requires the split learned-beta preset')
    if args.token_chord_gates and (not learned_beta or args.signed_chord_steps):
        parser.error('--token-chord-gates requires the positive split learned-beta preset')
    if args.token_beta_gate and (not learned_beta or args.token_chord_gates or args.signed_chord_steps):
        parser.error('--token-beta-gate requires the shared positive split learned-beta preset')
    geometry = args.chord_velocity_geometry
    if args.fixed_first_attention_mu and (not args.fuse_chord_read_norm or args.chord_velocity_write):
        parser.error('--fixed-first-attention-mu uses the fused token-only Chord recipe')
    if args.chord_velocity_write and (args.channel_chord_coefficients or not args.fuse_chord_read_norm):
        parser.error('--chord-velocity-write uses the fused scalar Chord default')
    if args.channel_chord_coefficients and not args.fuse_chord_read_norm:
        parser.error('--channel-chord-coefficients requires the fused default Chord control')
    if args.zero_init_velocity_embedding and (not args.token_only_velocity or not args.fuse_chord_read_norm
            or args.fixed_first_attention_mu or args.chord_velocity_write or args.channel_chord_coefficients):
        parser.error('--zero-init-velocity-embedding uses the standard fused scalar token-only Chord preset')
    if args.zero_velocity and (geometry != 'project_only' or not args.fuse_chord_read_norm
            or args.fixed_first_attention_mu or args.chord_velocity_write or args.channel_chord_coefficients):
        parser.error('--zero-velocity uses the standard scalar fused project-only Chord preset')
    if args.fuse_chord_read_norm and not (args.token_only_velocity or args.zero_velocity):
        parser.error('--fuse-chord-read-norm requires --token-only-velocity or --zero-velocity')
    if args.token_only_velocity and geometry != 'project_only':
        parser.error('--token-only-velocity requires --chord-velocity-geometry project_only')
    if geometry and (not learned_beta or args.token_beta_gate or args.token_chord_gates or args.signed_chord_steps):
        parser.error('--chord-velocity-geometry requires the shared positive split learned-beta preset')
    split_write = preset == 'ngpt_lookahead_split_write' or geodesic or learned_beta
    axis_write = preset == 'ngpt_lookahead_axis_write' or split_write
    if axis_write and not args.hopper_only:
        parser.error('The axis-write experiment is scheduled on full H100/H200 only')
    short_control = preset == 'ngpt_hbformer_no_wn'
    steps = 5000 if short_control or (mode == 'ngpt_lookahead' and not (full_lookahead or axis_write)) else STEPS
    budget = '5k-steps' if steps == 5000 else '10b'
    if args.fresh_10b:
        budget='10b-from-scratch'
    ROOT=HOME_ROOT/f'residual-nonzero-{preset}-{budget}-20261117'
    connection = preset in ('mhc', 'attnres')
    if connection and not args.hopper_only:
        parser.error('The matched connection runs use full H100/H200')
    if connection:
        ROOT = HOME_ROOT/f'residual-nonzero-{preset}-{budget}-20261120'
    if axis_write:
        tag = 'geodesic-split' if geodesic else 'split' if split_write else 'axis'
        if learned_beta:
            tag = 'split-learned-beta'
        if args.signed_chord_steps:
            tag = 'split-learned-beta-signed'
        ROOT=HOME_ROOT/f'residual-arc-{tag}-write-10b-20261119'
        if args.token_chord_gates:
            ROOT=HOME_ROOT/'residual-chord-token-alpha-gates-10b-20261120'
        if args.token_beta_gate:
            ROOT=HOME_ROOT/'residual-chord-token-beta-gate-10b-20261120'
        if geometry:
            ROOT=HOME_ROOT/f'residual-chord-velocity-{geometry.replace("_","-")}-10b-20261120'
    if args.token_only_velocity:
        ROOT=HOME_ROOT/'residual-chord-velocity-token-only-bf16-10b-20261120'
        if args.fuse_chord_read_norm:
            ROOT=HOME_ROOT/'residual-chord-fused-read-token-only-bf16-10b-20261120'
    if args.zero_init_velocity_embedding:
        ROOT=HOME_ROOT/'residual-chord-scalar-zero-init-velocity-embed-10b-20261121'
    if args.zero_velocity:
        ROOT=HOME_ROOT/'residual-chord-scalar-from-rest-10b-20261121'
    if args.chord_velocity_write:
        ROOT=HOME_ROOT/'residual-chord-scalar-rho-v-10b-20261121'
    if args.channel_chord_coefficients:
        ROOT=HOME_ROOT/'residual-chord-channel-alpha-beta-10b-20261121'
    if args.fixed_first_attention_mu:
        coefficients = 'channel' if args.channel_chord_coefficients else 'scalar'
        ROOT=HOME_ROOT/f'residual-chord-{coefficients}-fixed-first-attn-mu1-10b-20261121'
    if arc_optimizer:
        tag = 'arc' if preset == 'arc' else 'arc-muon-direction'
        if preset == 'arc_muon_radius':
            tag += '-radius-step'
        if args.no_arc_lookahead:
            tag += '-mu0'
        date_tag = '20261119' if args.no_arc_lookahead else '20261118'
        ROOT = HOME_ROOT/f'optimizer-{tag}-10b-{date_tag}'
    if args.hopper_only:
        ROOT = ROOT.with_name(ROOT.name + '-hopper-only')
    KEY=ROOT.name
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'submit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not any((ROOT/n).exists() for n in ('STOPPED','STOPPED.json'))
        if (ROOT/'state.json').exists():
            print((ROOT/'state.json').read_text());return
        assert not (ROOT/'submission-intent.json').exists(), 'Recover previous submission; do not duplicate'
        active=json.loads(command(['condor_q','yzhong239','-constraint',f'NgptTrainKey == "{KEY}"','-json']) or '[]')
        assert not active
        assert not command(['git','status','--porcelain'])
        commit=command(['git','rev-parse','HEAD'])
        transport=json.loads((REPO/'research/checkpoint-safety/validation.json').read_text())
        assert transport['complete'] and transport['actual_eviction']
        data=json.loads((HOME_ROOT/'datasets/fineweb-edu-10bt/current-packed-cache.json').read_text())
        assert data['ready'] and data['sequence_length']==2048
        assert data['manifest']['training_tokens']==STEPS*32768
        assert (HOME_ROOT/'.netrc').stat().st_mode&0o777==0o600
        for name in ('bundles','logs','results'):(ROOT/name).mkdir(exist_ok=True)
        assert not any((ROOT/'results').iterdir())
        bundle=ROOT/'bundles'
        command(['git','archive','--format=tar','--prefix=nanochat-src/','-o',str(bundle/'source.tar'),commit])
        wheel='transformer_engine_torch-2.20.2-cp311-cp311-linux_x86_64.whl'
        (bundle/wheel).symlink_to(ASSETS/'bundles'/wheel)
        shutil.copy2(REPO/'research/connections/ngpt/train.sh',ROOT/'train.sh')
        (ROOT/'train.sh').chmod(0o755)
        run_id=uuid.uuid4().hex[:8]
        source=dict(commit=commit,formal_training=True,plan=KEY,run_id=run_id,
            model_variant=preset,preset=preset,residual_dynamics=mode,velocity_embedding=mode in ('ngpt_heavy_ball','heavy_ball_rms','ngpt_hbformer','ngpt_lookahead','lookahead_only'),
            ngpt_alpha_init=.05,ngpt_force_rms_floor=1e-6,ngpt_velocity_mode='ambient',
            residual_projection_init='normal',residual_projection_init_std=.02,
            output_projection_actual_std=.005,momentum_beta_init=.5,momentum_gamma_init=.5,
            momentum_eta_init=.1 if mode=='ngpt_heavy_ball' else 1.,
            model=dict(depth=8,width=512,heads=4,sequence_length=2048,vocab_size=32768),steps=steps,training_tokens=steps*32768,batch_tokens=32768,microbatch=16,
            seed=42,initialization='seed42_from_scratch_no_diagnostic_weights',
            preflight='29 tests; target128 + other two modes32 updates each; initial unit targets; exact CLI resume; fresh seed42 formal',
            gate_timeout_seconds=2400,failure_debug_seconds=900,
            attention_backend='fa2',data_uri=data['data_uri'],data_archive_bytes=data['archive']['bytes'],
            dataset_revision=data['manifest']['dataset_revision'],tokenizer=data['manifest']['tokenizer']['tokenizer'],
            runtime='mhc-official-runtime-7546767.tar.gz + frozen uv.lock',
            checkpoint_seconds=1800,evaluate_test=True,
            scheduling='H100/H200 first; A100 after 3600 continuous idle seconds, with same-allocation nGPT preflight',
            passed_families=[],compatible_fallback_backend='A100 FA2/CCE runtime validated in skip-scale/validation.json',
            prior_h100_reference_step_seconds=.06704051233828068 if mode=='vanilla' else .0756970327347517,
            timing_reference='previous zero-init H100 diagnostics, estimate only; measure all three nonzero modes in preflight')
        if args.hopper_only:
            source.update(hopper_only=True, scheduling='Full H100/H200 only at startup and on resume')
            del source['compatible_fallback_backend']
        if connection:
            base = json.loads((HOME_ROOT/'residual-nonzero-vanilla-10b-20261117/summary.json').read_text())
            assert base['complete'] and base['run_id']=='2eac64db' and base['steps']==steps
            for key in ('model','steps','seed','microbatch','batch_tokens','attention_backend','dataset_revision',
                        'data_uri','residual_projection_init','residual_projection_init_std','output_projection_actual_std'):
                assert source[key]==base['source'][key], key
            if preset=='mhc':
                previous=json.loads((REPO/'research/connections/pretrain/results.json').read_text())['trials']['mhc']
            else:
                previous=json.loads((REPO/'research/connections/attnres/validation.json').read_text())['restart']
            source.update(connection_variant=preset, streams=4 if preset=='mhc' else 1,
                          matched_backbone_init=True, schedule_steps=STEPS, reference_run=base['run_id'],
                          preflight='Seed-matched backbone, finite gradients and 32 real FA2 steps with a step-12 restart',
                          reference_gpu=previous['gpu'], reference_step_seconds=previous['steady_step_seconds'],
                          timing_reference='Historical H200 zero-init timing with its original attention backend')
            (ROOT/'comparison-reference.json').write_text(json.dumps(dict(run_id=base['run_id'],
                dev_ce=base['final_dev_ce'], dev_tokens=base['dev_tokens'], step=steps), indent=2)+'\n')
        if arc_optimizer:
            base = json.loads((HOME_ROOT/'residual-nonzero-vanilla-10b-20261117/summary.json').read_text())
            assert base['complete'] and base['run_id'] == '2eac64db' and base['steps'] == steps
            for key in ('model', 'seed', 'batch_tokens', 'microbatch', 'residual_projection_init',
                        'residual_projection_init_std', 'dataset_revision', 'residual_dynamics'):
                assert source[key] == base['source'][key], key
            source.update(matrix_optimizer='arc', arc_mu=1., arc_beta=.5, schedule_steps=STEPS,
                          arc_radius='initial matrix Frobenius norm', arc_step='matrix_lr * sqrt(fan_out)',
                          reference_run=base['run_id'], reference_initial_dev_ce=base['initial_dev_ce'],
                          preflight='Analytical update and restart, accumulated query gradients, production CLI exit85, matched Muon speed',
                          timing_reference='Vanilla reference only; same-card Arc timing measured in preflight')
            if preset in ('arc_muon', 'arc_muon_radius'):
                source.update(arc_direction='muon', comparison_steps=1000,
                              preflight='Same-GPU 1000-step Muon, raw Arc and Muon-direction Arc; independent restart check',
                              direction_recipe='native scheduled Muon momentum, row equilibration, Polar Express and factored variance; no matrix decay')
            if preset == 'arc_muon_radius':
                source.update(arc_step_scale='radius', arc_step='matrix_lr * initial_frobenius_radius',
                              preflight='Same-GPU 1000-step Muon and Muon-direction Arc with fan-out versus radius steps; independent restart check',
                              prior_h100_reference_step_seconds=.07567,
                              timing_reference='Previous fan-out Arc + Muon direction, median H100 steps 101–1000')
            if args.no_arc_lookahead:
                source.update(arc_mu=0., lookahead_reference_run='4042e49e',
                              preflight='Matched 1000-step radius-step Muon directions at mu=1 and mu=0; mu=0 checkpoint restart')
            (ROOT/'comparison-reference.json').write_text(json.dumps(dict(run_id=base['run_id'],
                dev_ce=base['final_dev_ce'], dev_tokens=base['dev_tokens'], step=steps), indent=2)+'\n')
        if mode in ('ngpt_lookahead', 'lookahead_only'):
            del source['momentum_gamma_init'], source['momentum_eta_init']
            source.update(velocity_init='token_position', velocity_embedding_std=.02, momentum_beta_trainable=False,
                          lookahead_mu_init=0., ngpt_velocity_mode='lookahead_transport' if mode=='ngpt_lookahead' else 'ambient', ngpt_weight_norm=False, schedule_steps=STEPS,
                          reference_run='d065c132', backbone_weight_constraint='none',
                          preflight='Zero-gate equality with nGPT, trainable lookahead, tangent geometry, causal KV, gradients, exact resume and same-GPU speed')
        if (args.channel_chord_coefficients or args.chord_velocity_write or args.fixed_first_attention_mu
                or args.zero_velocity or args.zero_init_velocity_embedding):
            control_dir = HOME_ROOT/'residual-chord-fused-read-token-only-bf16-10b-20261120-hopper-only'
            control_id = 'a7163d23'
            if args.fixed_first_attention_mu and args.channel_chord_coefficients:
                control_dir = HOME_ROOT/'residual-chord-channel-alpha-beta-10b-20261121-hopper-only'
                control_id = '9a7fd31d'
            control = json.loads((control_dir/'bundles/source.json').read_text())
            assert control['run_id']==control_id and control['fuse_chord_read_norm'] and not control['recompute_chord_force']
            assert control['seed']==42 and control['steps']==steps and control['chord_state_dtype']=='bfloat16'
            assert control['velocity_init']=='token' and control['chord_velocity_geometry']==geometry
            source = dict(control, plan=KEY, commit=commit, run_id=run_id, reference_run=control['run_id'],
                          paired_run=control['run_id'])
            if args.zero_init_velocity_embedding:
                source.update(velocity_embedding_std=0.,
                              preflight='Skipped by request; start formal training at step zero with seed 42')
            elif args.zero_velocity:
                source.update(velocity_init='zero', velocity_embedding=False, initial_velocity_projected=False,
                              preflight='Skipped by request; start formal training at step zero with seed 42')
            elif args.fixed_first_attention_mu:
                source.update(fixed_first_attention_mu=True,
                              preflight='Fixed first read, embedding gradients, compiled KV, exact restart and 128-step profile; independent seed42 formal')
            elif args.chord_velocity_write:
                source.update(chord_velocity_write=True, rho_parameterization='tanh', rho_init=0.,
                              preflight='Zero-rho equality and drift, FP64 VJP, compiled cache, exact resume; paired scalar and rho-v 128-step profiles; fresh seed42 formal')
            else:
                source.update(channel_chord_coefficients=True,
                              preflight='Channel VJP, compiled cache and exact optimizer resume; matched scalar and channel 128-step profiles; fresh seed42 formal')
        elif args.token_only_velocity:
            control_dir = HOME_ROOT/'residual-chord-velocity-project-only-10b-20261120-hopper-only'
            control = json.loads((control_dir/'bundles/source.json').read_text())
            assert control['run_id']=='af1c3266' and control['velocity_init']=='token_position'
            assert control['chord_velocity_geometry']==geometry and control['initial_velocity_projected']
            for key in ('model','preset','seed','initialization','steps','schedule_steps','batch_tokens',
                        'microbatch','data_uri','dataset_revision','tokenizer','runtime','attention_backend','hopper_only'):
                assert control[key] == source[key], key
            source = dict(control, plan=KEY, commit=commit, run_id=run_id, velocity_init='token', chord_state_dtype='bfloat16',
                          reference_run=control['run_id'], paired_run=control['run_id'],
                          preflight='BF16 h/v storage, long-sequence memory, gradients, cache and restart; 128 steps each for BF16 token-only, FP32 token-only and BF16 token plus position',
                          timing_reference='Historical H200 timing of transported Chord; paired project-only timing in preflight')
            if args.fuse_chord_read_norm:
                source.update(fuse_chord_read_norm=True, recompute_chord_force=False,
                              preflight='Fused RMSNorm and projection remat VJPs, compiled cache and resume; 128 same-card steps each for separate, fused, fused+remat; fresh seed42 formal uses fusion only')
            else:
                (ROOT/'comparison-reference.json').write_text(json.dumps(dict(run_id=control['run_id'],
                    summary_path=str(control_dir/'summary.json'), step=steps), indent=2)+'\n')
        elif axis_write:
            reference = HOME_ROOT/'residual-nonzero-ngpt_lookahead-10b-from-scratch-20261117'
            base = json.loads((reference/'summary.json').read_text())
            assert base['complete'] and base['run_id'] == 'd4293868' and base['source']['ngpt_velocity_mode'] == 'lookahead_transport'
            for key in ('model','seed','training_tokens','microbatch','batch_tokens','schedule_steps','data_uri',
                        'dataset_revision','tokenizer','attention_backend','residual_dynamics','residual_projection_init',
                        'residual_projection_init_std','ngpt_force_rms_floor','ngpt_weight_norm','ngpt_alpha_init',
                        'lookahead_mu_init','momentum_beta_init','velocity_init','velocity_embedding_std'):
                assert source[key] == base['source'][key], key
            source.update(reference_run=base['run_id'], reference_step_seconds=base['steady_step_seconds'],
                          reference_gpu=base['gpu'], timing_reference='Prior Arc Residual; matched control timed during preflight')
            if split_write:
                control_tag = 'split' if geodesic or learned_beta else 'axis'
                paired = json.loads((HOME_ROOT/f'residual-arc-{control_tag}-write-10b-20261119-hopper-only/bundles/source.json').read_text())
                assert paired['ngpt_velocity_mode'] == f'lookahead_{control_tag}_write' and paired['reference_run'] == base['run_id']
                source.update(ngpt_velocity_mode=preset.removeprefix('ngpt_'), paired_run=paired['run_id'],
                              parallel_step_init=.05, perpendicular_step_init=.05, step_parameterization='independent_softplus',
                              preflight='Equal-step parity, independent gradients, KV cache, restart; paired 128 steps versus alpha/kappa on the training GPU')
                if geodesic:
                    source.update(residual_write='split sphere log then sphere exp',
                                  velocity_observation='parallel transport of actual arc step and incoming velocity along selected arc',
                                  preflight='SLERP, oriented arc transport, degeneracies, gradients, KV cache, restart; paired 128 steps versus chord split')
                if learned_beta:
                    assert paired['momentum_beta_init'] == .5 and not paired['momentum_beta_trainable']
                    source.update(momentum_beta_trainable=True, beta_parameterization='sigmoid',
                                  learned_beta_sites=2*source['model']['depth']-1,
                                  preflight='Fixed-beta init parity, beta gradients, causal KV, optimizer restart; paired 128 steps versus fixed-beta chord split')
                    if args.signed_chord_steps or args.token_chord_gates or args.token_beta_gate or geometry:
                        control_dir = HOME_ROOT/'residual-arc-split-learned-beta-write-10b-20261119-hopper-only'
                        control = json.loads((control_dir/'bundles/source.json').read_text())
                        for key in ('model','seed','training_tokens','batch_tokens','microbatch','dataset_revision',
                                    'schedule_steps','ngpt_velocity_mode','momentum_beta_init','learned_beta_sites',
                                    'velocity_init','parallel_step_init','perpendicular_step_init','step_parameterization'):
                            assert control[key] == source[key], key
                        source['paired_run'] = control['run_id']
                        if args.signed_chord_steps:
                            source.update(signed_chord_steps=True, step_parameterization='independent_softplus_with_linear_lower_extension',
                                          preflight='Both signs, matched initial values and derivatives, causal KV, optimizer restart; paired 128 steps versus positive steps with learned beta')
                        else:
                            base = json.loads((control_dir/'summary.json').read_text())
                            assert base['complete'] and base['run_id'] == control['run_id'] and base['steps'] == steps
                            source.update(reference_run=control['run_id'],
                                          reference_step_seconds=base['steady_step_seconds'],reference_gpu=base['gpu'],
                                          residual_kernel='fused_chord_triton')
                            if geometry:
                                source.update(chord_velocity_geometry=geometry,
                                              initial_velocity_projected=geometry=='project_only',
                                              preflight='Independent velocity formula and VJPs; matched 128 steps against transported Chord, restart at 32')
                            else:
                                source.update(gate_slope_init=0.,
                                              preflight='Fused/eager VJPs at active gates, neutral init, KV, exact resume; paired 128 steps versus positive learned-beta control')
                                if args.token_beta_gate:
                                    source.update(token_beta_gate=True, gate_signal='cosine(transported_velocity,tangent_realized_chord)',
                                                  gate_scale='sigmoid(raw_beta+w*c)')
                                else:
                                    source.update(token_chord_gates=True, gate_signal='cosine(target-h, velocity)',
                                                  gate_scale='2*sigmoid(w*c)')
            else:
                archive = reference/'results'/f"train-{json.loads((reference/'state.json').read_text())['job']}-checkpoints.tar"
                with tarfile.open(archive,'r:') as tar:
                    member = tar.getmember('checkpoints/best.pt')
                    assert member.isfile() and member.size == base['checkpoint_files']['best.pt']['bytes']
                (bundle/'axis-reference.tar').symlink_to(archive)
                source.update(ngpt_velocity_mode='lookahead_axis_write', kappa_init=0., reference_best_step=base['best_dev_step'],
                              reference_best_dev_ce=base['best_dev_ce'],
                              preflight='Read trained Arc Residual dev alignment; zero-kappa parity, geometry, gradients, cache, recovery; paired 128-step runs on the training GPU')
            (ROOT/'comparison-reference.json').write_text(json.dumps(dict(run_id=base['run_id'],
                dev_ce=base['final_dev_ce'], dev_tokens=base['dev_tokens'], step=base['steps']), indent=2)+'\n')
        if mode == 'lookahead_only':
            source.update(reference_run='2eac64db', combined_reference_run='d4293868',
                          residual_write='vanilla_addition', velocity_observation='realized_displacement',
                          preflight='Zero-gate parity with vanilla, finite active-gate gradients, KV cache, exact resume; 128 steps and matched vanilla/combined timing')
        if mode=='ngpt_hbformer':
            source.pop('output_projection_actual_std')
            source.update(velocity_init='token_position', velocity_embedding_std=.02, momentum_eta_init=1.,
                          momentum_eta_trainable=False, momentum_gamma_init=math.log(2.), ngpt_velocity_mode='transport_rms',
                          momentum_velocity_rms_floor=.1, backbone_weight_constraint='ngpt_unit_l2_init_and_post_step',
                          output_projection_std_before_normalization=.005,
                          preflight='HBFormer velocity/step, unit weight axes and post-step projection, cache, gradients, exact resume; target128 and two controls32',
                          timing_reference='prior H100 diagnostics, excludes new post-step weight normalization overhead')
        if short_control or preset == 'ngpt_hbformer_velocity_ln':
            source.update(ngpt_weight_norm=False, backbone_weight_constraint='none', schedule_steps=STEPS,
                          preflight='Unprojected initialization and optimizer, same-card 128 steps, cache/gradients and exact CLI resume',
                          timing_reference='prior H100 residual diagnostics, confirm same-card speed in preflight')
        if preset == 'ngpt_hbformer_velocity_ln':
            source.update(ngpt_velocity_mode='transport_layernorm', velocity_layernorm_eps=1e-5,
                          velocity_gain_init=1., velocity_layernorm_bias=False,
                          preflight='Affine velocity LayerNorm before projection, geometric and gradient tests, exact resume; target128, vanilla and matched RMS 32 each')
        if mode=='heavy_ball_rms':
            source.update(velocity_init='token', velocity_embedding_std=.02,
                          momentum_eta_init=.05, momentum_velocity_rms_floor=.1,
                          preflight='9 Euclidean velocity RMS tests; target128, vanilla and transport_rms 32 each; exact CLI resume',
                          timing_reference='estimate from prior H100 residual diagnostics; measure same GPU in preflight')
        if preset=='ngpt_momentum_transport_rms':
            report_path=HOME_ROOT/'ngpt-transport-rms-debug-20261117/validation.json'
            report=json.loads(report_path.read_text())
            assert report['complete'] and report['exit_code']==0 and report['resume']['complete']
            assert report['tests']==dict(tests=57, failures=0, errors=0, skipped=0)
            profile=report['profile']
            target=profile['models'][mode]
            assert profile['complete'] and profile['gpu']=='NVIDIA H200' and target['steps']==128
            validated=report['source']['commit']
            changes=command(['git','diff','--name-only',validated,commit]).splitlines()
            assert all(path.startswith('research/connections/ngpt/') for path in changes), changes
            source.update(ngpt_velocity_mode='transport_rms',momentum_velocity_rms_floor=.1,
                momentum_eta_init=.05,velocity_init='token',velocity_embedding_std=.02,
                validation_job=report['job'],validated_model_commit=validated,passed_families=['sm90'],
                preflight='reuse verified sm90 model and runtime; full transport checks on sm80',
                reference_step_seconds=target['median_step_seconds'],reference_gpu=profile['gpu'],
                timing_reference='transport_rms 16x2048 same-card H200 median over steps 17-32')
            del source['prior_h100_reference_step_seconds']
            config=target['config']
            for key in ('residual_dynamics','ngpt_velocity_mode','momentum_velocity_rms_floor',
                        'velocity_init','velocity_embedding_std','ngpt_force_rms_floor','ngpt_alpha_init',
                        'momentum_eta_init','momentum_beta_init','momentum_gamma_init',
                        'residual_projection_init','residual_projection_init_std'):
                assert config[key]==source[key], key
            assert all(config[key]==value for key,value in source['model'].items())
            assert profile['seed']==source['seed'] and profile['microbatch']==source['microbatch']
            assert profile['batch_tokens']==source['batch_tokens'] and profile['attention_backend']=='fa2'
            shutil.copy2(report_path,bundle/'transport-validation.json')
        if full_lookahead:
            parent=HOME_ROOT/f'residual-nonzero-{preset}-5k-steps-20261117'
            summary=json.loads((parent/'summary.json').read_text())
            previous=json.loads((parent/'state.json').read_text())
            old=summary['source']
            assert previous['status']=='completed' and summary['complete'] and summary['steps']==5000
            assert previous['source']==old and summary['run_id']==previous['run_id']==old['run_id']
            assert not json.loads(command(['condor_q',previous['job'],'-json']) or '[]')
            assert old['schedule_steps']==steps and old['steps']==summary['steps']
            metadata={'plan','commit','steps','training_tokens','run_id'}
            if args.hopper_only:
                metadata.update({'hopper_only', 'scheduling', 'compatible_fallback_backend'})
            assert {k:v for k,v in old.items() if k not in metadata} == {k:v for k,v in source.items() if k not in metadata}
            changed=command(['git','diff','--name-only',old['commit'],commit]).splitlines()
            assert all(p.startswith('research/connections/ngpt/') or p=='scripts/connection_train.py' for p in changed),changed
            source['reference_step_seconds']=summary['steady_step_seconds']
            source['reference_gpu']=summary['gpu']
        if args.fresh_10b:
            assert run_id!=summary['run_id']
            gate=parent/'ngpt-sm90.json'
            report=json.loads(gate.read_text())
            assert report['complete'] and report['source']==old and report['resume']['complete']
            source.update(validation_job=previous['job'],validated_model_commit=old['commit'])
            shutil.copy2(gate,bundle/'lookahead-validation.json')
        if args.extend_to_schedule:
            run_id=source['run_id']=summary['run_id']
            archives=parent/'results'
            checkpoints=archives/f"train-{previous['job']}-checkpoints.tar"
            with tarfile.open(checkpoints,'r:') as tar:
                for name in ('latest.pt','best.pt'):
                    member=tar.getmember('checkpoints/'+name)
                    assert member.isfile() and member.size==summary['checkpoint_files'][name]['bytes']
            (bundle/'parent-checkpoints.tar').symlink_to(checkpoints)
            (bundle/'parent-results.tar').symlink_to(archives/f"train-{previous['job']}.tar")
            source['continuation']=dict(source=old,job=previous['job'],step=summary['steps'],dev_ce=summary['final_dev_ce'],
                                        checkpoint_files=summary['checkpoint_files'])
        (bundle/'source.json').write_text(json.dumps(source,indent=2)+'\n')
        inputs=[str(p.relative_to(ROOT)) for p in bundle.iterdir()]
        inputs += [str(HOME_ROOT/'.netrc'),data['data_uri'],
            'osdf:///chtc/staging/y/yzhong239/mhc-official-runtime-7546767.tar.gz',
            'https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-compat-13-1_590.48.01-1_amd64.deb']
        lines=[]
        batch_name = f'nanochat-{tag}-10b' if arc_optimizer else f'nanochat-nonzero-residual-{budget}-{preset}'
        job_args=f'{run_id} {mode}'
        if source['ngpt_velocity_mode'] != 'ambient':
            job_args+=' '+source['ngpt_velocity_mode']
        if args.signed_chord_steps:
            job_args+=' --signed-chord-steps'
            batch_name+='-signed-steps'
        if args.token_chord_gates:
            job_args+=' --token-chord-gates'
            batch_name+='-token-alpha-gates'
        if args.token_beta_gate:
            job_args+=' --token-beta-gate'
            batch_name+='-token-beta-gate'
        if geometry:
            job_args+=f' --chord-velocity-geometry={geometry}'
            batch_name+='-velocity-'+geometry.replace('_','-')
        if args.zero_velocity:
            job_args+=' --zero-velocity'
            batch_name+='-from-rest'
        if args.token_only_velocity:
            job_args+=' --token-only-velocity'
            batch_name+='-token-only-bf16'
        if args.fuse_chord_read_norm:
            job_args+=' --fuse-chord-read-norm'
            batch_name+='-fused-read'
        if args.zero_init_velocity_embedding:
            job_args+=' --zero-init-velocity-embedding'
            batch_name+='-zero-init-velocity-embed'
        if args.chord_velocity_write:
            job_args+=' --chord-velocity-write'
            batch_name+='-rho-v'
        if args.channel_chord_coefficients:
            job_args+=' --channel-chord-coefficients'
            batch_name+='-channel-alpha-beta'
        if args.fixed_first_attention_mu:
            job_args+=' --fixed-first-attention-mu'
            batch_name+='-fixed-first-attn-mu1'
        if connection:
            job_args += ' ambient --' + preset
        if arc_optimizer:
            job_args += ' ambient --' + preset.replace('_', '-')
        if args.no_arc_lookahead:
            job_args+=' --no-arc-lookahead'
        if args.fresh_10b:
            job_args+=' --fresh-10b'
        if args.extend_to_schedule:
            job_args+=' --extend-to-schedule'
        if mode == 'ngpt_hbformer' and source.get('ngpt_weight_norm') is False:
            job_args+=' --no-ngpt-weight-norm'
        for line in (ASSETS/'gate.sub').read_text().splitlines():
            if line.startswith('executable ='):line=f'executable = train.sh\narguments = {job_args}'
            elif line.startswith('request_disk ='):line='request_disk = 80GB'
            elif line.startswith('gpus_minimum_capability ='):
                line=f'gpus_minimum_capability = {9.0 if args.hopper_only else 8.0}'
            elif line.startswith('require_gpus ='):line='require_gpus = '+require_gpus()
            elif line.startswith('rank ='):line='rank = '+RANK
            elif line.startswith('transfer_input_files ='):line='transfer_input_files = '+','.join(inputs)
            elif line.startswith('transfer_output_files ='):line='transfer_output_files = results.tar,checkpoints.tar'
            elif line.startswith('transfer_output_remaps ='):
                line='transfer_output_remaps = "results.tar=results/train-$(Cluster).$(Process).tar;checkpoints.tar=results/train-$(Cluster).$(Process)-checkpoints.tar"'
            elif line.startswith('queue '):continue
            lines.append(line.replace('logs/stream.','logs/train.').replace('nanochat-depth-diagnostic',batch_name)
                .replace('DepthDiagnosticKey','NgptTrainKey').replace('depth-diagnostic-20261103',KEY))
        lines += ['checkpoint_exit_code = 85','transfer_checkpoint_files = checkpoints.tar,results.tar',
            'encrypt_input_files = /home/yzhong239/.netrc','+InteractiveJob = true',
            'on_exit_hold = (ExitBySignal == true) || ((ExitCode != 0) && (ExitCode != 85))',
            'hold = true','queue 1']
        (ROOT/'train.sub').write_text('\n'.join(lines)+'\n')
        command(['condor_submit','-dry-run','dry-run.ad','train.sub'],ROOT)
        state=dict(status='submitting',run_id=run_id,source=source,formal_training=True,submitted_at=time.time(),
            wandb_url=f'https://wandb.ai/yibozhong657-none/nanochat/runs/{run_id}')
        if 'reference_step_seconds' in source:
            state['estimated_h200_training_hours']=(steps-source.get('continuation',{}).get('step',0))*source['reference_step_seconds']/3600
        else:
            state['estimated_h100_training_hours']=steps*source['prior_h100_reference_step_seconds']/3600
        (ROOT/'submission-intent.json').write_text(json.dumps(state,indent=2)+'\n')
        job=command(['condor_submit','-terse','train.sub'],ROOT).split()[0]
        state.update(job=job,status='held_for_audit')
        (ROOT/'state.json').write_text(json.dumps(state,indent=2)+'\n')
        ad=json.loads(command(['condor_q',job,'-json']))[0]
        assert ad['JobStatus']==5 and ad.get('NumJobStarts',0)==0
        assert ad['RequestGPUs']==1 and ad['RequestCpus']==2 and ad['RequestMemory']==16384
        assert ad['RequestDisk']==80*1024**2 and ad['GPUsMinCapability']==(9.0 if args.hopper_only else 8.0)
        assert 'H100|H200' in ad['RequireGPUs'] and 'A100' not in ad['RequireGPUs']
        assert 'nanochat-uv-v4.sif' in ad['TransferInput'] and data['data_uri'] in ad['TransferInput']
        assert 'gate-data.tar' not in ad['TransferInput'] and '.netrc' in ad['EncryptInputFiles']
        if preset=='ngpt_momentum_transport_rms':
            assert 'transport-validation.json' in ad['TransferInput']
        assert ad['SuccessCheckpointExitCode']==85 and set(ad['TransferCheckpoint'].split(','))=={'results.tar','checkpoints.tar'}
        assert set(ad['TransferOutput'].split(','))=={'results.tar','checkpoints.tar'}
        assert ad['NgptTrainKey']==KEY and ad['InteractiveJob'] and ad['WantGPULab']
        assert ad['Args']==job_args
        if axis_write:
            assert ('axis-reference.tar' in ad['TransferInput']) == (not split_write)
            assert 'parent-' not in ad['TransferInput'] and 'continuation' not in source
        if mode == 'lookahead_only' or arc_optimizer or connection:
            assert not any(name in ad['TransferInput'] for name in ('parent-', 'checkpoints.tar', 'results.tar'))
        if args.fresh_10b:
            assert 'lookahead-validation.json' in ad['TransferInput'] and 'continuation' not in source
            assert not any(x in ad['TransferInput'] for x in ('checkpoints.tar','results.tar','.pt,'))
        if args.extend_to_schedule:
            assert 'parent-checkpoints.tar' in ad['TransferInput'] and 'parent-results.tar' in ad['TransferInput']
        assert 'backfill' in ad['Rank'] and 'countMatches' in ad['Rank']
        keys=('ClusterId','ProcId','JobStatus','Args','RequestGPUs','RequestCpus','RequestMemory','RequestDisk',
            'RequireGPUs','TransferInput','TransferOutput','TransferCheckpoint','SuccessCheckpointExitCode',
            'EncryptInputFiles','ContainerImage','Rank','OnExitHold')
        (ROOT/'submission-verified.json').write_text(json.dumps({k:ad.get(k) for k in keys},indent=2)+'\n')
        command(['condor_release',job])
        state['status']='training_submitted'
        (ROOT/'state.json').write_text(json.dumps(state,indent=2)+'\n')
        print(json.dumps({k:v for k,v in state.items() if k in ('status','job','run_id','wandb_url') or k.startswith('estimated_')},indent=2))


if __name__=='__main__':main()
