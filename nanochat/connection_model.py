"""Full-attention vanilla and NVIDIA mHC on nanochat attention and MLP.

This research model keeps native GPT and its SSSL/extras path unchanged.
The residual option matches the audited 58.72M full-attention vanilla baseline.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from nanochat.gpt import Block, GPTConfig, Linear, norm
from nanochat.nvidia_connections import NvidiaHyperConnection, NvidiaHyperHead, expand_streams, check_nvidia_device
from nanochat.optim import MuonAdamW

CONNECTION_VARIANTS = ('baseline', 'mhc', 'mhc_no_sinkhorn', 'mhc_channel_no_sinkhorn', 'attnres', 'scaled_hc', 'scaled_hc_all', 'skip_scale', 'attnres_scale')


@dataclass
class ConnectionConfig:
    variant: str = 'baseline'
    depth: int = 8
    width: int = 512
    heads: int = 4
    sequence_length: int = 2048
    vocab_size: int = 32768
    streams: int = 4
    sinkhorn_iterations: int = 20
    activation_checkpoint: bool = False
    connection_backend: str = 'nvidia'
    fused_connection_grad: bool = False
    architecture: str = 'vanilla'
    window_pattern: str = 'SSSL'  # native only; vanilla is always full attention
    attention_kernel: str = 'original'
    compact_routing: bool = False
    attnres_block_size: int = 2  # Moonshot paper's S: Attention and MLP each count once
    attnres_block_size_unit: str = 'sublayers'
    latent_feedback: bool = False
    feedback_eval_passes: int = 4
    feedback_source: str = 'top'
    feedback_attention_backend: str = 'triton'
    residual_dynamics: str = 'vanilla'
    velocity_init: str | None = None
    velocity_embedding_std: float = 0.02
    momentum_beta_init: float = 0.5
    momentum_gamma_init: float | None = None
    momentum_eta_init: float | None = None
    normalize_minimal_force: bool = False
    ngpt_force_rms_floor: float = 0.1
    ngpt_alpha_init: float = 0.05
    lookahead_mu_init: float = 0.0
    fixed_first_attention_mu: bool = False
    ngpt_weight_norm: bool | None = None
    ngpt_velocity_mode: str | None = None
    momentum_velocity_rms_floor: float = 0.1
    residual_projection_init: str | None = None
    residual_projection_init_std: float = 0.02
    signed_chord_steps: bool = False
    token_chord_gates: bool = False
    token_beta_gate: bool = False
    channel_chord_coefficients: bool = False
    chord_velocity_write: bool = False
    chord_six_controls: str = 'off'
    chord_velocity_geometry: str = 'parallel_transport'
    chord_state_dtype: str | None = None
    fuse_chord_read_norm: bool = False
    recompute_chord_force: bool = False
    match_velocity_table_with_ffn: bool = False
    chord_attnres: bool = False
    chord_attnres_backend: str = 'direct'
    chord_history_correction: bool = False
    chord_history_updates: bool = False
    chord_query_only_lookahead: bool = False
    native_chord_write: str | None = None
    native_chord_control_lr: float | None = None
    chord_update_based: bool = False
    chord_soft_force_tau: float = 0.0

    def __post_init__(self):
        if self.native_chord_control_lr is not None and (
                self.native_chord_write is None or not math.isfinite(self.native_chord_control_lr)
                or self.native_chord_control_lr <= 0):
            raise ValueError('A native Chord control LR requires native Chord and a positive finite peak LR')
        hbformer = self.residual_mode == 'ngpt_hbformer'
        lookahead = self.residual_mode in ('ngpt_lookahead', 'lookahead_only')
        if self.fixed_first_attention_mu and self.residual_mode != 'ngpt_lookahead':
            raise ValueError('Fixed first Attention lookahead requires spherical lookahead')
        if not math.isfinite(self.lookahead_mu_init):
            raise ValueError('Lookahead must start at a finite value')
        if self.ngpt_weight_norm is None:
            self.ngpt_weight_norm = hbformer and self.ngpt_velocity_mode != 'transport_layernorm'
        if self.ngpt_weight_norm and not hbformer:
            raise ValueError('nGPT weight projection is supported with ngpt_hbformer')
        if self.residual_projection_init is None:
            self.residual_projection_init = 'normal' if hbformer or lookahead else 'zero'
        if self.residual_projection_init not in ('zero', 'normal'):
            raise ValueError('Unknown residual output projection initialization')
        if not math.isfinite(self.residual_projection_init_std) or self.residual_projection_init_std <= 0:
            raise ValueError('Residual output projection initialization std must be positive and finite')
        if self.residual_projection_init != 'zero' and (self.variant not in ('baseline', 'mhc', 'attnres') or self.latent_feedback):
            raise ValueError('Random output initialization requires baseline, mHC, or vanilla AttnRes')
        if self.residual_dynamics not in ('vanilla', 'heavy_ball', 'heavy_ball_rms', 'ngpt_heavy_ball', 'ngpt_hbformer', 'ngpt', 'ngpt_lookahead', 'lookahead_only'):
            raise ValueError('Unknown residual dynamics')
        if not math.isfinite(self.ngpt_alpha_init) or not 0 < self.ngpt_alpha_init < 1:
            raise ValueError('nGPT alpha must start strictly between zero and one')
        if self.momentum_gamma_init is None:
            self.momentum_gamma_init = math.log(2.) if hbformer else 0.5
        if self.ngpt_velocity_mode is None:
            if self.residual_mode == 'ngpt_lookahead':
                self.ngpt_velocity_mode = 'lookahead_transport'
            else:
                self.ngpt_velocity_mode = 'transport_rms' if self.residual_mode == 'ngpt_heavy_ball' or hbformer else 'ambient'
        if self.momentum_eta_init is None:
            if self.residual_mode == 'ngpt_heavy_ball':
                self.momentum_eta_init = 0.05 if self.ngpt_velocity_mode == 'transport_rms' else 0.1
            else:
                self.momentum_eta_init = 0.05 if self.residual_mode == 'heavy_ball_rms' else 1.0
        if self.ngpt_velocity_mode == 'transport_layernorm' and not hbformer:
            raise ValueError('Transported velocity LayerNorm requires the unit-step HBFormer')
        if self.residual_mode == 'lookahead_only' and self.ngpt_velocity_mode != 'ambient':
            raise ValueError('Additive lookahead uses Euclidean velocity')
        if self.chord_six_controls not in ('off', 'static', 'small_dynamic', 'learned_dynamic'):
            raise ValueError('Unknown six-coefficient Chord setting')
        if self.chord_six_controls != 'off' and (self.architecture != 'vanilla'
                or self.ngpt_velocity_mode != 'lookahead_split_learned_beta' or not self.fuse_chord_read_norm
                or self.chord_velocity_geometry != 'project_only' or self.recompute_chord_force
                or self.channel_chord_coefficients or self.chord_velocity_write or self.chord_attnres
                or self.chord_query_only_lookahead or self.fixed_first_attention_mu):
            raise ValueError('Six-coefficient Chord requires vanilla BF16 scalar project-only updates')
        if not math.isfinite(self.chord_soft_force_tau) or self.chord_soft_force_tau < 0:
            raise ValueError('Soft force tau must be finite and nonnegative')
        token_update = self.chord_update_based and self.dynamic_chord_controls
        if token_update and not self.chord_soft_force_tau:
            raise ValueError('Token-conditioned update-based Chord requires a positive soft force tau')
        if self.chord_soft_force_tau and not token_update and (
                self.architecture != 'vanilla' or self.residual_mode != 'ngpt_lookahead'
                or self.ngpt_velocity_mode != 'lookahead_split_learned_beta' or self.chord_six_controls != 'off'
                or self.token_chord_gates or self.token_beta_gate or self.channel_chord_coefficients or self.chord_velocity_write):
            raise ValueError('Soft force requires scalar split Chord or a token-conditioned update-based write')
        if self.chord_update_based and (
                self.architecture != 'vanilla' or self.ngpt_velocity_mode != 'lookahead_split_learned_beta'
                or self.chord_six_controls == 'static' or self.channel_chord_coefficients
                or self.token_chord_gates or self.token_beta_gate or self.chord_velocity_write):
            raise ValueError('Update-based Chord requires vanilla split steps with scalar retention or h-only controls')
        if self.normalize_minimal_force and self.ngpt_velocity_mode != 'lookahead_minimal':
            raise ValueError('Normalized no-alpha writes require minimal Chord')
        simple_chord = self.ngpt_velocity_mode in ('lookahead_token_single', 'lookahead_minimal')
        if simple_chord and (self.architecture != 'vanilla' or self.chord_velocity_geometry != 'project_only'
                or not self.fuse_chord_read_norm or self.recompute_chord_force or self.channel_chord_coefficients
                or self.chord_attnres or self.chord_query_only_lookahead or self.chord_velocity_write
                or self.fixed_first_attention_mu):
            raise ValueError('This Chord mode uses vanilla BF16 fused project-only updates')
        spherical_lookahead = simple_chord or self.split_chord_write or self.ngpt_velocity_mode in ('lookahead_transport', 'lookahead_axis_write', 'lookahead_geodesic_split')
        if self.signed_chord_steps and not self.split_chord_write:
            raise ValueError('Signed component steps require a split chord write')
        if self.token_chord_gates and (self.ngpt_velocity_mode != 'lookahead_split_learned_beta' or self.signed_chord_steps):
            raise ValueError('Token chord gates require positive split steps and learned velocity decay')
        if self.chord_state_dtype is None:
            self.chord_state_dtype = 'bfloat16' if self.chord_velocity_geometry == 'project_only' else 'float32'
        if (self.fuse_chord_read_norm or self.recompute_chord_force) and (
                self.chord_state_dtype != 'bfloat16' or self.chord_velocity_geometry != 'project_only'):
            raise ValueError('Fused read/norm and force recomputation require BF16 project-only Chord')
        if self.chord_state_dtype not in ('bfloat16', 'float32'):
            raise ValueError('Unknown Chord state dtype')
        if self.chord_state_dtype == 'bfloat16' and (self.residual_mode != 'ngpt_lookahead'
                or self.ngpt_velocity_mode == 'lookahead_geodesic_split'):
            raise ValueError('BF16 Chord state storage requires spherical lookahead with a chord write')
        if self.chord_velocity_geometry not in ('parallel_transport', 'project_only', 'ambient'):
            raise ValueError('Unknown Chord velocity geometry')
        if not simple_chord and self.chord_velocity_geometry != 'parallel_transport' and (
                self.ngpt_velocity_mode != 'lookahead_split_learned_beta' or self.signed_chord_steps
                or self.token_chord_gates or self.token_beta_gate):
            raise ValueError('Velocity geometry ablations require positive split steps and layerwise beta')
        if self.chord_velocity_write and (self.chord_velocity_geometry != 'project_only' or self.channel_chord_coefficients):
            raise ValueError('Direct velocity write requires scalar project-only Chord')
        if self.channel_chord_coefficients and (
                self.ngpt_velocity_mode != 'lookahead_split_learned_beta' or self.chord_velocity_geometry != 'project_only'):
            raise ValueError('Channel alpha and beta require the project-only learned-beta Chord write')
        if self.token_beta_gate and (self.ngpt_velocity_mode != 'lookahead_split_learned_beta'
                                     or self.signed_chord_steps or self.token_chord_gates):
            raise ValueError('Token beta gate requires shared positive split steps and learned velocity decay')
        if (self.residual_mode == 'ngpt_lookahead') != spherical_lookahead:
            raise ValueError('Spherical lookahead requires a compatible Chord or geodesic velocity mode')
        if not spherical_lookahead and self.ngpt_velocity_mode not in ('ambient', 'transport_rms', 'transport_layernorm'):
            raise ValueError('Unknown nGPT velocity mode')
        if not math.isfinite(self.momentum_velocity_rms_floor) or self.momentum_velocity_rms_floor <= 0:
            raise ValueError('Velocity RMS floor must be positive and finite')
        if not math.isfinite(self.ngpt_force_rms_floor) or self.ngpt_force_rms_floor <= 0:
            raise ValueError('Normalized-target force RMS floor must be positive and finite')
        if self.velocity_init is None:
            if hbformer or lookahead:
                self.velocity_init = 'token_position'
            else:
                self.velocity_init = 'token' if self.residual_mode in ('heavy_ball', 'heavy_ball_rms', 'ngpt_heavy_ball') else 'zero'
        if hbformer and (self.velocity_init != 'token_position' or self.ngpt_velocity_mode == 'ambient'
                         or self.momentum_eta_init != 1 or self.residual_projection_init != 'normal'):
            raise ValueError('ngpt_hbformer requires token_position velocity, parallel transport, fixed unit step and nonzero projections')
        if self.residual_mode == 'lookahead_only' and self.velocity_init != 'token_position':
            raise ValueError('Additive lookahead requires token and position velocity embeddings')
        if self.velocity_init == 'token_position' and not (hbformer or lookahead):
            raise ValueError('Token plus position velocity requires HBFormer or lookahead')
        if self.velocity_init not in ('zero', 'token', 'token_position', 'projected_h0'):
            raise ValueError('Unknown velocity initialization')
        if self.velocity_init == 'projected_h0' and (self.architecture != 'vanilla'
                or self.residual_mode != 'ngpt_lookahead' or self.chord_velocity_geometry == 'ambient'):
            raise ValueError('Projected h0 velocity requires vanilla spherical lookahead with tangent velocity')
        if self.velocity_init == 'token' and self.residual_mode not in ('heavy_ball', 'heavy_ball_rms', 'ngpt_heavy_ball', 'ngpt_lookahead'):
            raise ValueError('Token-only velocity requires heavy ball or spherical lookahead')
        if not math.isfinite(self.velocity_embedding_std) or self.velocity_embedding_std < 0:
            raise ValueError('Velocity embedding initialization must be nonnegative and finite')
        if not (math.isfinite(self.momentum_beta_init) and 0 < self.momentum_beta_init < 1):
            raise ValueError('Residual momentum beta must start strictly between zero and one')
        if any(not math.isfinite(x) or x <= 0 for x in (self.momentum_gamma_init, self.momentum_eta_init)):
            raise ValueError('Residual momentum gamma and eta must start positive and finite')
        if self.native_chord_write == 'token_bounded' and abs(self.lookahead_mu_init) >= 3:
            raise ValueError('Bounded Chord lookahead must start strictly between -3 and 3')
        if self.native_chord_write in ('token_vnorm', 'token_bounded', 'token_h') and not self.fuse_chord_read_norm:
            raise ValueError('Native token-conditioned Chord uses the fused preview/norm read')
        if self.native_chord_write is not None:
            if (self.native_chord_write not in ('tied', 'three_scalar', 'token_vnorm', 'token_bounded', 'token_h') or self.architecture != 'native'
                    or self.residual_mode != 'ngpt_lookahead' or self.ngpt_velocity_mode != 'lookahead_split_learned_beta'
                    or self.chord_velocity_geometry != 'project_only' or self.velocity_init != 'token'
                    or self.chord_state_dtype != 'bfloat16' or self.streams != 1 or self.recompute_chord_force
                    or self.channel_chord_coefficients or self.chord_velocity_write or self.chord_attnres
                    or self.chord_query_only_lookahead or self.match_velocity_table_with_ffn or self.fixed_first_attention_mu):
                raise ValueError('Native Chord uses the scalar BF16 project-only token-velocity configuration')
        if self.residual_mode != 'vanilla' and ((self.architecture != 'vanilla' and self.native_chord_write is None) or self.variant != 'baseline'
                or self.latent_feedback or self.attention_kernel != 'original' or self.fused_connection_grad):
            raise ValueError('Residual dynamics require the vanilla baseline and original attention')
        if self.feedback_source not in ('top','layers') or self.feedback_attention_backend not in ('torch','triton'):
            raise ValueError('Unknown feedback source or attention backend')
        if self.feedback_source == 'layers' and not self.latent_feedback:
            raise ValueError('Layer feedback requires latent_feedback')
        if type(self.feedback_eval_passes) is not int or self.feedback_eval_passes < 1:
            raise ValueError('Feedback evaluation passes must be a positive integer')
        if self.latent_feedback and (self.architecture != 'vanilla' or self.variant != 'baseline'
                                    or self.attention_kernel != 'original'):
            raise ValueError('Latent feedback uses the vanilla baseline and original attention')
        if self.variant in ('skip_scale','attnres_scale'):
            if self.architecture != 'vanilla' or self.streams != 1 or self.fused_connection_grad:
                raise ValueError('Skip scale uses vanilla, exactly one residual stream, and ordinary gradients')
        if self.attnres_block_size_unit != 'sublayers':
            raise ValueError('AttnRes block size uses sublayers; load legacy metadata with from_checkpoint')
        if type(self.attnres_block_size) is not int or self.attnres_block_size < 2 or self.attnres_block_size % 2:
            raise ValueError('AttnRes block size counts Attention/MLP sublayers and must be even and >=2; Full AttnRes (S=1) is not implemented')
        if self.variant in ('attnres','attnres_scale'):
            if self.width > 16384:
                raise ValueError('NVIDIA AttnRes supports hidden width at most16384')
            if self.architecture != 'vanilla' or self.fused_connection_grad:
                raise ValueError('Initial AttnRes port uses vanilla and its own NVIDIA autograd')
            if math.ceil(self.depth / self.attnres_block_layers) > 16:
                raise ValueError('NVIDIA AttnRes supports at most16 completed history entries')
        if self.compact_routing and self.variant != 'mhc_channel_no_sinkhorn':
            raise ValueError('Compact routing applies only to the8-coefficient channel connector')
        if self.attention_kernel not in ('original','fla_triton'):
            raise ValueError('Unknown attention kernel')
        if self.attention_kernel == 'fla_triton' and (self.architecture != 'vanilla' or self.variant != 'baseline'):
            raise ValueError('Initial FLA port supports vanilla baseline backbone only')
        if self.architecture not in ('vanilla', 'native'):
            raise ValueError(f'Unknown backbone architecture: {self.architecture}')
        if not self.window_pattern or any(c not in 'SL' for c in self.window_pattern):
            raise ValueError('window_pattern must contain S and/or L')
        if self.connection_backend not in ('nvidia', 'nvidia_precise'):
            raise ValueError('Only the full NVIDIA mHC implementation is supported')
        if self.variant not in CONNECTION_VARIANTS:
            raise ValueError(f'Unknown connection variant: {self.variant}')
        if self.variant == 'mhc_channel_no_sinkhorn' and (self.architecture != 'vanilla' or self.fused_connection_grad):
            raise ValueError('Local channel mixing initially supports vanilla and unfused gradient accumulation')
        if self.variant in ('scaled_hc','scaled_hc_all') and (self.architecture != 'vanilla' or self.fused_connection_grad):
            raise ValueError('Scaled HC supports vanilla and ordinary gradient accumulation')
        if self.sinkhorn_iterations != 20:
            raise ValueError('The NVIDIA composite API requires 20 Sinkhorn iterations')
        if min(self.depth, self.width, self.heads, self.sequence_length, self.vocab_size, self.streams) < 1:
            raise ValueError('Model dimensions must be positive')
        if self.width % self.heads or (self.width // self.heads) % 2:
            raise ValueError('RoPE requires an even integral head dimension')
        if self.chord_query_only_lookahead and (self.residual_mode != 'ngpt_lookahead' or self.chord_attnres):
            raise ValueError('Query-only lookahead requires spherical Chord without depth attention')
        if self.chord_history_updates and not self.chord_history_correction:
            raise ValueError('Block chord values require the additive history read')
        if self.chord_history_correction and not self.chord_attnres:
            raise ValueError('A Chord history correction requires ChordAttnRes')
        if self.chord_attnres:
            if (self.residual_mode != 'ngpt_lookahead' or self.chord_velocity_geometry != 'project_only'
                    or self.chord_state_dtype != 'bfloat16' or self.recompute_chord_force):
                raise ValueError('ChordAttnRes requires BF16 project-only Chord with ordinary force writes')
            if self.width not in (128, 512) or (self.depth-1)//self.attnres_block_layers > 16:
                raise ValueError('Endpoint kernels support widths 128/512 and at most 16 historical endpoints')
        if self.chord_attnres_backend not in ('direct', 'packed') or (not self.chord_attnres and self.chord_attnres_backend != 'direct'):
            raise ValueError('Unknown or inapplicable ChordAttnRes kernel')
        if self.match_velocity_table_with_ffn:
            if self.residual_mode != 'ngpt_lookahead' or self.velocity_init != 'zero':
                raise ValueError('Matching the velocity table with FFN capacity requires Chord from rest')
            if self.vocab_size % (2 * self.depth):
                raise ValueError('Velocity table parameters must divide evenly between all FFN input/output pairs')
        if self.sinkhorn_iterations < 1:
            raise ValueError('Sinkhorn needs at least one iteration')
        if self.variant not in ('baseline', 'attnres', 'attnres_scale', 'skip_scale') and (self.streams != 4 or self.width % 128):
            raise ValueError('NVIDIA mHC kernels require four streams and width divisible by 128')

    @property
    def ffn_width(self):
        extra = self.vocab_size // (2 * self.depth) if self.match_velocity_table_with_ffn else 0
        return 4 * self.width + extra

    @property
    def dynamic_chord_controls(self):
        return self.chord_six_controls in ('small_dynamic', 'learned_dynamic')

    @property
    def split_chord_write(self):
        return self.ngpt_velocity_mode in ('lookahead_split_write', 'lookahead_split_learned_beta')

    @property
    def residual_mode(self):
        return self.residual_dynamics

    @property
    def attnres_block_layers(self):
        """Complete Transformer layers per block, for NeMo's layer-boundary recurrence."""
        return self.attnres_block_size // 2

    @classmethod
    def from_checkpoint(cls, config):
        """Read pre-cleanup NVIDIA/vanilla metadata; reject retired model weights."""
        config = dict(config)
        if config.pop('riemannian_momentum', False) or config.get('residual_dynamics') in ('riemannian', 'sphere'):
            raise ValueError('Checkpoint belongs to the retired sphere residual route; use its archived code')
        # Historical checkpoints predating this field used v0=0. Preserve their
        # architecture while newly constructed momentum models default to tokens.
        config.setdefault('velocity_init', 'zero')
        config.setdefault('chord_state_dtype', 'float32')
        # Never reinterpret an ambient checkpoint as tangent/normalized momentum.
        config.setdefault('ngpt_velocity_mode', 'ambient')
        if config.pop('vocab_gate', False):
            raise ValueError('Checkpoint belongs to the retired vocabulary gate experiment')
        for name in ('vocab_gate_alpha', 'vocab_gate_backend', 'vocab_gate_chunk'):
            config.pop(name, None)
        if config.pop('ffn_readout_width', 0):
            raise ValueError('Checkpoint belongs to the retired FFN readout experiment')
        config.pop('ffn_readout_backend', None)
        config.pop('ffn_readout_init', None)
        if config.pop('attention_mode', 'full') != 'full':
            raise ValueError('Checkpoint belongs to the retired HCA experiment')
        for name in list(config):
            if name.startswith('hca_'):
                config.pop(name)
        if config.pop('window_final_full', True) is not True:
            raise ValueError('Checkpoint belongs to the retired native frontloaded-attention experiment')
        if config.pop('attention_pattern', 'full') != 'full':
            raise ValueError('Checkpoint belongs to the retired dilated-attention experiment')
        for name in ('sparse_budget', 'sparse_core', 'sparse_sinks', 'sparse_backend'):
            config.pop(name, None)  # inert defaults stored by retained full-attention models
        config.pop('kalman_formula', None)  # retired route; other variants stored this inert default
        if config.pop('mlp_memory', 'none') != 'none':
            raise ValueError('Checkpoint belongs to the retired play-memory experiment')
        config.pop('play_radius_init', None)
        # Metadata written before the unit marker counted complete Transformer
        # layers. Convert once, without changing grouping or checkpoint tensors.
        unit = config.get('attnres_block_size_unit', 'transformer_layers')
        if unit == 'transformer_layers':
            size = config.get('attnres_block_size', 1)
            if type(size) is not int or size < 1:
                raise ValueError('Invalid legacy AttnRes block size')
            config['attnres_block_size'] = 2 * size
            config['attnres_block_size_unit'] = 'sublayers'
        if config.pop('attention_statistic', 'none') != 'none':
            raise ValueError('Checkpoint belongs to a retired attention-statistics experiment')
        config.pop('moment_query_chunk', None)
        if config.pop('swa_looping', False):
            raise ValueError('Checkpoint belongs to the retired SWA looping experiment')
        for name in ('swa_loops', 'swa_window'):
            config.pop(name, None)
        if config.pop('attention_looping', False):
            raise ValueError('Checkpoint belongs to the retired attention looping experiment')
        for name in ('attention_loops', 'attention_loop_window'):
            config.pop(name, None)
        if config.pop('attention_output_ns', False):
            raise ValueError('Checkpoint belongs to the retired attention output NS experiment')
        if config.get('variant', 'baseline') not in CONNECTION_VARIANTS:
            raise ValueError('Checkpoint belongs to a retired model variant')
        if config.pop('stream_mlp_kernel', 'legacy') != 'legacy':
            raise ValueError('Checkpoint belongs to the retired stream MLP experiment')
        # Legacy metadata was inert on the retained native/vanilla/mHC models.
        for name in ("chunk_size", "lookback_enabled", "lookback_update",
                     "recompute_chunk_size", "recompute_previous", "recompute_input",
                     "recompute_memory"):
            config.pop(name, None)
        projection = config.pop('projection_backend', 'official')
        if projection != 'official':
            raise ValueError('Projection-only checkpoints belong to a retired implementation')
        if config.get('variant', 'baseline') == 'baseline':
            config['connection_backend'] = 'nvidia'  # unused by the vanilla model
        return cls(**config)


class ConnectionBlock(Block):
    def __init__(self, native, index, config):
        super().__init__(native, index)
        self.attn.ve_gate = None
        self.variant = config.variant
        self.residual_mode = config.residual_mode
        self.fuse_chord_read_norm = config.fuse_chord_read_norm
        self.recompute_chord_force = config.recompute_chord_force
        self.chord_query_only_lookahead = config.chord_query_only_lookahead
        self.token_controls = config.ngpt_velocity_mode == 'lookahead_token_single' or config.dynamic_chord_controls
        self.compiled_attention = self.compiled_mlp = None
        if config.residual_mode != 'vanilla':
            from nanochat.momentum import residual_update
            self.attn_residual = residual_update(config, fixed_unit_mu=index == 0 and config.fixed_first_attention_mu)
            self.mlp_residual = residual_update(config, velocity_consumed=index + 1 < config.depth)
        if self.variant == 'skip_scale':
            self.attn_skip_scale = nn.Parameter(torch.ones(config.width))
            self.mlp_skip_scale = nn.Parameter(torch.ones(config.width))
        elif self.variant != 'baseline':
            connection = NvidiaHyperConnection
            if self.variant == 'mhc_channel_no_sinkhorn':
                from nanochat.channel_connections import ChannelHyperConnection
                connection = ChannelHyperConnection
            if self.variant == 'scaled_hc':
                from nanochat.scaled_connections import ScaledHyperConnection
                connection = ScaledHyperConnection
            if self.variant == 'scaled_hc_all':
                from nanochat.scaled_all_connections import AllScaledHyperConnection
                connection = AllScaledHyperConnection
            kwargs = dict(use_tf32=config.connection_backend == 'nvidia', fused_grad_acc=config.fused_connection_grad,
                          sinkhorn=config.variant != 'mhc_no_sinkhorn')
            if config.variant == 'mhc_channel_no_sinkhorn':
                kwargs['compact_routing']=config.compact_routing
            self.attn_connection = connection(config.width, config.streams, 2*index, config.sinkhorn_iterations, **kwargs)
            self.mlp_connection = connection(config.width, config.streams, 2*index+1, config.sinkhorn_iterations, **kwargs)

    def attention_branch(self, h, cos, sin, kv_cache=None):
        return self.attn(norm(h), None, (cos, sin), (-1, 0), kv_cache)

    def mlp_branch(self, h):
        return self.mlp(norm(h))

    def momentum_forward(self, h, v, cos, sin, kv_cache=None, *, attn_update=None, mlp_update=None):
        if self.token_controls:
            q, controls = self.attn_residual.read(h, v)
            attn = self.compiled_attention if kv_cache is None else None
            force = (attn or self.attn)(q, None, (cos, sin), (-1, 0), kv_cache)
            h, v = self.attn_residual.write(h, v, force, controls)
            q, controls = self.mlp_residual.read(h, v)
            force = (self.compiled_mlp or self.mlp)(q)
            return self.mlp_residual.write(h, v, force, controls)
        if self.fuse_chord_read_norm or self.recompute_chord_force or self.chord_query_only_lookahead:
            remat = self.recompute_chord_force and self.training and torch.is_grad_enabled() and kv_cache is None
            q = self.attn_residual.branch_input(h, v, cos.dtype, self.fuse_chord_read_norm)
            # Share the current state's pre-norm across K and V. Only Q anticipates motion.
            kv_input = norm(h.to(cos.dtype)) if self.chord_query_only_lookahead else None
            features = self.attn.output_features(q, None, (cos, sin), (-1, 0), kv_cache, kv_input=kv_input)
            h, v = self.attn_residual.projected_write(h, v, features, self.attn.c_proj, remat)
            q = self.mlp_residual.branch_input(h, v, cos.dtype, self.fuse_chord_read_norm)
            return self.mlp_residual.projected_write(h, v, self.mlp.output_features(q), self.mlp.c_proj, remat)
        lookahead = self.residual_mode in ('ngpt_lookahead', 'lookahead_only')
        query = self.attn_residual.query(h, v) if lookahead else h
        force = self.attention_branch(query.to(cos.dtype), cos, sin, kv_cache)
        h, v = (attn_update or self.attn_residual)(h, v, force)
        query = self.mlp_residual.query(h, v) if lookahead else h
        force = self.mlp_branch(query.to(cos.dtype))
        return (mlp_update or self.mlp_residual)(h, v, force)

    def forward(self, x, cos, sin, kv_cache=None):
        if self.residual_mode == 'ngpt':
            force = self.attention_branch(x.to(cos.dtype), cos, sin, kv_cache)
            x = self.attn_residual(x, force)
            return self.mlp_residual(x, self.mlp_branch(x.to(cos.dtype)))
        if self.variant == 'baseline':
            return super().forward(x, None, (cos, sin), (-1, 0), kv_cache)
        if self.variant == 'skip_scale':
            from nanochat.skip_scale import scaled_skip
            x = scaled_skip(x, self.attention_branch(x, cos, sin, kv_cache), self.attn_skip_scale)
            return scaled_skip(x, self.mlp_branch(x), self.mlp_skip_scale)
        h, skip, post = self.attn_connection.read(x)
        y = (self.compiled_attention or self.attention_branch)(h, cos, sin, kv_cache)
        x = self.attn_connection.write(y, skip, post)
        h, skip, post = self.mlp_connection.read(x)
        y = (self.compiled_mlp or self.mlp_branch)(h)
        return self.mlp_connection.write(y, skip, post)


def _init_ffn_weights(up, down, config):
    bound = math.sqrt(3 / config.width) * .4
    nn.init.uniform_(up, -bound, bound)
    if config.residual_projection_init == 'normal':
        nn.init.normal_(down, std=config.residual_projection_init_std / math.sqrt(2 * config.depth))
    else:
        nn.init.zeros_(down)


@torch.no_grad()
def _widen_chord_ffns(blocks, config):
    # Grow after the ordinary initialization so every shared parameter, including
    # the original FFN channels, has the same seed-matched value as the control.
    for block in blocks:
        mlp = block.mlp
        n = mlp.c_fc.out_features
        up = Linear(config.width, config.ffn_width, bias=False)
        down = Linear(config.ffn_width, config.width, bias=False)
        up.weight[:n].copy_(mlp.c_fc.weight)
        down.weight[:, :n].copy_(mlp.c_proj.weight)
        _init_ffn_weights(up.weight[n:], down.weight[:, n:], config)
        mlp.c_fc, mlp.c_proj = up, down


class ConnectionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.residual_mode != 'vanilla' and type(self) is ConnectionModel:
            raise ValueError('Use build_connection_model for momentum residuals')
        if config.variant in ('attnres','attnres_scale'):
            raise ValueError('Use build_connection_model for AttnRes')
        if config.architecture != 'vanilla':
            raise ValueError('Use build_connection_model for native nanochat')
        self.config = config
        native = GPTConfig(sequence_len=config.sequence_length, vocab_size=config.vocab_size,
            n_layer=config.depth, n_head=config.heads, n_kv_head=config.heads,
            n_embd=config.width, window_pattern='L')
        self.wte = nn.Embedding(config.vocab_size, config.width)
        self.blocks = nn.ModuleList([ConnectionBlock(native, i, config) for i in range(config.depth)])
        self.lm_head = Linear(config.width, config.vocab_size, bias=False)
        nn.init.normal_(self.wte.weight, std=.8)
        nn.init.normal_(self.lm_head.weight, std=.001)
        s = math.sqrt(3 / config.width)
        for b in self.blocks:
            for p in (b.attn.c_q.weight, b.attn.c_k.weight, b.attn.c_v.weight):
                nn.init.uniform_(p, -s, s)
            if config.residual_projection_init == 'normal':
                nn.init.normal_(b.attn.c_proj.weight, std=config.residual_projection_init_std/math.sqrt(2*config.depth))
            else:
                nn.init.zeros_(b.attn.c_proj.weight)
            _init_ffn_weights(b.mlp.c_fc.weight, b.mlp.c_proj.weight, config)
        if config.match_velocity_table_with_ffn:
            _widen_chord_ffns(self.blocks, config)
        if config.variant not in ('baseline', 'skip_scale'):
            for b in self.blocks:
                b.attn_connection.reset_parameters(); b.mlp_connection.reset_parameters()
        if config.variant in ('baseline', 'skip_scale'):
            self.stream_head = None
        else:
            head_class=NvidiaHyperHead
            if config.variant=='scaled_hc_all':
                from nanochat.scaled_all_connections import ScaledHyperHead
                head_class=ScaledHyperHead
            self.stream_head = head_class(config.width, config.streams, use_tf32=config.connection_backend == 'nvidia')
        if self.stream_head is not None:
            self.stream_head.reset_parameters()
        d = config.width // config.heads
        inv = 1 / (100000 ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
        freq = torch.outer(torch.arange(config.sequence_length, dtype=torch.float32), inv)
        self.register_buffer('cos', freq.cos()[None, :, None, :], persistent=False)
        self.register_buffer('sin', freq.sin()[None, :, None, :], persistent=False)
        self.compiled_steps = None
        if config.attention_kernel == 'fla_triton':
            from nanochat.fla_attention import FlaAttention
            for block in self.blocks:
                block.attn = FlaAttention(block.attn)

    def prepare(self, device='cuda', dtype=torch.bfloat16, compile_blocks=False):
        if self.config.variant not in ('baseline', 'skip_scale'):
            check_nvidia_device(device)
            if dtype != torch.bfloat16:
                raise ValueError('NVIDIA mHC kernels require BF16 activations')
        self.to(device)
        self.wte.to(dtype=dtype)
        self.cos = self.cos.to(dtype); self.sin = self.sin.to(dtype)
        if compile_blocks and self.config.variant in ('baseline', 'skip_scale'):
            self.compiled_steps = [torch.compile(b.forward, dynamic=False, fullgraph=True) for b in self.blocks]
        elif compile_blocks:
            # NVIDIA TE supplies its own kernels and custom backward.
            # Compile only the unchanged attention/MLP branches with Inductor.
            for b in self.blocks:
                b.compiled_attention = torch.compile(b.attention_branch, dynamic=False, fullgraph=True)
                b.compiled_mlp = torch.compile(b.mlp_branch, dynamic=False, fullgraph=True)
        return self

    def hidden(self, idx, kv_cache=None):
        return norm(self.backbone(norm(self.wte(idx)), kv_cache))

    def backbone(self, x, kv_cache=None, return_layers=False):
        """Unnormalized top state, also used by detached latent feedback passes."""
        start = 0 if kv_cache is None else kv_cache.get_pos()
        if start + x.shape[1] > self.config.sequence_length:
            raise ValueError('Input exceeds configured context length')
        if self.config.variant not in ('baseline', 'skip_scale'):
            x = expand_streams(x, self.config.streams)
        cos, sin = self.cos[:, start:start+x.shape[1]], self.sin[:, start:start+x.shape[1]]
        states = [] if return_layers else None
        for i, b in enumerate(self.blocks):
            # Cache positions mutate across calls and stay outside static
            # fullgraph training capture; attention keeps its GPU kernels.
            fn = self.compiled_steps[i] if self.compiled_steps is not None and kv_cache is None else b.forward
            if self.config.activation_checkpoint and self.training and torch.is_grad_enabled():
                if kv_cache is not None:
                    raise ValueError('Activation checkpointing cannot mutate a KV cache')
                x = checkpoint(fn, x, cos, sin, use_reentrant=False)
            else:
                x = fn(x, cos, sin, kv_cache)
            if return_layers:
                states.append(x)
        if self.stream_head is not None:
            x = self.stream_head(x)
        return (x, torch.stack(states, dim=2)) if return_layers else x

    def forward(self, idx, targets=None, dense=False, kv_cache=None, loss_reduction="mean"):
        h = self.hidden(idx, kv_cache)
        return self.readout(h, targets, dense, loss_reduction)

    def readout(self, h, targets=None, dense=False, loss_reduction="mean"):
        w = self.lm_head.weight.to(h.dtype)
        if targets is None or dense:
            logits = 15 * torch.tanh(F.linear(h, w).float() / 15)
            if targets is None:
                return logits
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction=loss_reduction)
            return loss.view_as(targets) if loss_reduction == "none" else loss
        from cut_cross_entropy import linear_cross_entropy
        return linear_cross_entropy(h, w, targets, softcap=15., shift=0,
            reduction=loss_reduction, filter_eps=None, filter_e_grad=False, filter_c_grad=False,
            accum_e_fp32=True, accum_c_fp32=True)


def optimizer_for(model, batch_tokens=32768, *, matrix_optimizer='muon', arc_mu=1., arc_beta=.5, arc_direction='raw',
                  arc_step_scale='fan_out', velocity_embedding_lr=None):
    """Same backbone native LR rules; connector parameters get separate AdamW."""
    if matrix_optimizer != 'arc' and arc_step_scale != 'fan_out':
        raise ValueError('Selecting an Arc step scale requires the Arc matrix optimizer')
    if matrix_optimizer != 'arc' and arc_direction != 'raw':
        raise ValueError('Selecting an Arc direction requires the Arc matrix optimizer')
    if matrix_optimizer not in ('muon', 'arc'):
        raise ValueError(f'Unknown matrix optimizer {matrix_optimizer}')
    if matrix_optimizer == 'arc' and (model.config.architecture != 'vanilla' or
            model.config.variant != 'baseline' or model.config.residual_mode != 'vanilla'
            or model.config.latent_feedback or model.config.ngpt_weight_norm):
        raise ValueError('Arc pilot requires the vanilla baseline')
    velocity_table = getattr(model, 'velocity_embedding', None)
    if velocity_embedding_lr is not None:
        if velocity_table is None or model.config.architecture == 'native':
            raise ValueError('A velocity embedding LR requires a token velocity table')
        if not math.isfinite(velocity_embedding_lr) or velocity_embedding_lr <= 0:
            raise ValueError('The velocity embedding peak LR must be positive and finite')
    if model.config.architecture == 'native':
        return model.setup_optimizer(batch_tokens)
    c = model.config
    batch_scale = math.sqrt(batch_tokens / 2**19)
    width_scale = (c.width / 768) ** -.5
    # Use the baseline decay schedule for the parameter-matched FFN comparison.
    scaling_params = 12*c.depth*c.width*c.width + c.vocab_size*c.width
    ref_scaling_params = 12*12*768*768 + c.vocab_size*768
    wd = .28*batch_scale*(ref_scaling_params/scaling_params)
    groups = [
        dict(kind='adamw', params=[model.lm_head.weight], lr=.008*batch_scale*width_scale,
             betas=(.8, .96), eps=1e-10, weight_decay=.01),
        dict(kind='adamw', params=[model.wte.weight], lr=.3*batch_scale*width_scale,
             betas=(.8, .995), eps=1e-10, weight_decay=.001)]
    if velocity_table is not None:
        if velocity_embedding_lr is None:
            groups[1]['params'].append(velocity_table.weight)
        else:
            groups.append(dict(groups[1], params=[velocity_table.weight], lr=velocity_embedding_lr,
                               name='velocity_token_embedding'))
    if getattr(model, 'velocity_position_embedding', None) is not None:
        groups[1]['params'].append(model.velocity_position_embedding.weight)
    matrices, routing = [], []
    if getattr(model, 'velocity_projection', None) is not None:
        matrices.append(model.velocity_projection.weight)
    for b in model.blocks:
        for name, p in b.named_parameters():
            (routing if '_connection.' in name or name.startswith(('attn_residual.', 'mlp_residual.'))
                or name in ('attn_skip_scale', 'mlp_skip_scale') else matrices).append(p)
    if model.stream_head is not None:
        routing.extend(model.stream_head.parameters())
    if c.variant in ('attnres','attnres_scale') or c.chord_attnres:
        routing.extend(model.residual_parameters())
    if routing:
        groups.append(dict(kind='adamw', params=routing, lr=.008*batch_scale*width_scale,
                           betas=(.8, .96), eps=1e-10, weight_decay=0.))
    for shape in sorted({p.shape for p in matrices}):
        groups.append(dict(kind='muon', params=[p for p in matrices if p.shape == shape],
            lr=.02*batch_scale, momentum=.95, ns_steps=5, beta2=.9, weight_decay=wd))
    if c.latent_feedback:
        # Own group: single-pass updates must not advance momentum or decay W.
        feedback_params=[model.fusion.weight]
        if c.feedback_source == 'layers':
            feedback_params.append(model.feedback_query.weight)
        groups.append(dict(kind='muon', params=feedback_params, conditional=True,
            lr=.02*batch_scale, momentum=.95, ns_steps=5, beta2=.9, weight_decay=wd))
    ids = [id(p) for g in groups for p in g['params']]
    assert len(ids) == len(set(ids)) and set(ids) == {id(p) for p in model.parameters()}
    if c.residual_mode != 'vanilla' or c.variant in ('mhc_channel_no_sinkhorn', 'scaled_hc', 'scaled_hc_all', 'attnres_scale'):
        # Like native.setup_optimizer: the unchanged compiled AdamW function
        # specializes once per routing shape/dtype. Local generators introduce
        # two extra tensor ranks; this is a compiler cache budget, not an LR or
        # optimizer change, and must be set before the first optimizer step.
        from torch._dynamo import config as compiler_config
        signatures = {(tuple(p.shape), p.dtype) for g in groups if g['kind'] == 'adamw' for p in g['params']}
        compiler_config.recompile_limit = max(compiler_config.recompile_limit, 2*len(signatures))
    if matrix_optimizer == 'arc':
        from nanochat.arc import ArcAdamW
        opt = ArcAdamW(groups, mu=arc_mu, beta=arc_beta, direction=arc_direction, step_scale=arc_step_scale)
    else:
        opt = MuonAdamW(groups)
    if c.ngpt_weight_norm:
        opt.register_step_post_hook(lambda opt, args, kwargs: model.normalize_backbone_weights())
    for g in opt.param_groups:
        g['initial_lr'] = g['lr']
    return opt, wd


def build_connection_model(config):
    if config.architecture == 'native':
        from nanochat.native_connection_model import NativeConnectionModel
        return NativeConnectionModel(config)
    if config.chord_attnres:
        from nanochat.chord_attnres import ChordAttnResModel
        return ChordAttnResModel(config)
    if config.residual_mode in ('ngpt_lookahead', 'lookahead_only'):
        from nanochat.lookahead import LookaheadModel
        return LookaheadModel(config)
    if config.residual_mode == 'ngpt':
        from nanochat.momentum import NormalizedResidualModel
        return NormalizedResidualModel(config)
    if config.residual_mode != 'vanilla':
        from nanochat.momentum import MomentumModel
        return MomentumModel(config)
    if config.latent_feedback:
        from nanochat.latent_feedback import LatentFeedbackModel
        return LatentFeedbackModel(config)
    if config.variant == 'attnres_scale':
        from nanochat.scaled_attention_residuals import ScaledAttnResModel
        return ScaledAttnResModel(config)
    if config.variant == 'attnres':
        from nanochat.attention_residuals import AttnResModel
        return AttnResModel(config)
    return ConnectionModel(config)


def schedule(opt, step, total_steps, wd):
    down = round(.65*total_steps)
    if step < 40:
        lr_mult = (step+1)/40
    elif step <= total_steps-down:
        lr_mult = 1.
    else:
        progress = (total_steps-step)/down
        lr_mult = progress+(1-progress)*.05
    if step < 400:
        momentum = .85+(step/400)*(.97-.85)
    elif step >= total_steps-down:
        momentum = .97-(step-(total_steps-down))/down*.07
    else:
        momentum = .97
    for g in opt.param_groups:
        g['lr'] = g['initial_lr']*lr_mult
        if g['kind'] == 'muon' or g.get('direction') == 'muon':
            g['momentum'] = momentum
        if g['kind'] == 'muon':
            g['weight_decay'] = wd*.5*(1+math.cos(math.pi*step/total_steps))
    return lr_mult
