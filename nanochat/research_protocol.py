"""Identity and exact comparison fields for packed pretraining studies."""
import re


def validate_identity(group, name, baseline):
    if not isinstance(group, str) or not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*', group):
        raise ValueError('Provide a research question as --wandb-group, e.g. chord-initial-velocity')
    if not isinstance(name, str) or not name.strip() or any(ord(c) < 32 for c in name):
        raise ValueError('Provide a descriptive --run-name')
    if not isinstance(baseline, str) or not re.fullmatch(r'[a-zA-Z0-9_-]+', baseline):
        raise ValueError('Provide --baseline-run self or a run ID from nanochat-1b')
    return group, name.strip(), baseline


def comparison_protocol(config):
    """Quality at equal training tokens. Read identifiers from the saved manifest."""
    m, p, data = (config[k] for k in ('model', 'protocol', 'data_manifest'))
    return dict(version='packed-ntp-v1', dataset=data['dataset_revision'],
        train=data['files']['train']['sha256'], dev=data['files']['dev']['sha256'],
        tokenizer=data['tokenizer']['files']['tokenizer.pkl']['sha256'],
        seed=p['seed'], steps=p['steps'], schedule_steps=p['schedule_steps'],
        batch_tokens=p['batch_tokens'], microbatch=p['microbatch'], grad_accum=p['grad_accum'],
        schedule='warmup40-cooldown65pct-final5pct-v1', evaluation='full-dev-token-mean-ce-v1',
        depth=m['depth'], width=m['width'], heads=m['heads'], vocab=m['vocab_size'],
        architecture=m['architecture'], context=m['sequence_length'],
        windows=m['window_pattern'] if m['architecture']=='native' else 'full',
        attention=config['actual_attention_backend'], precision=config['training_dtype'],
        output_init=m['residual_projection_init'], output_init_std=m['residual_projection_init_std'],
        matched_backbone_init=p['matched_backbone_init'], loss=config['loss'],
        backbone_optimizer_groups=config['backbone_optimizer_groups'], matrix_optimizer=p.get('matrix_optimizer','muon'))
