"""Explicitly authorized runs; each preset owns one immutable 10B run ID."""
EXPERIMENTS = {
    'riemannian_zero': dict(mode='riemannian', velocity_init='zero',
        root='riemannian-momentum-10b-20261116'),
    'riemannian_embed': dict(mode='riemannian', velocity_init='token',
        root='riemannian-velocity-embedding-10b-20261116'),
    'heavy_ball_embed': dict(mode='heavy_ball', velocity_init='token',
        root='heavy-ball-velocity-embedding-10b-20261116'),
    'heavy_ball_zero': dict(mode='heavy_ball', velocity_init='zero',
        root='heavy-ball-zero-10b-20261116'),
    'sphere': dict(mode='sphere', velocity_init='zero',
        root='sphere-only-10b-20261116'),
}


def cli_args(name):
    c = EXPERIMENTS[name]
    return ['--residual-dynamics', c['mode'], '--velocity-init', c['velocity_init'],
            '--velocity-embedding-std', '.02']


if __name__ == '__main__':
    import sys
    print('\n'.join(cli_args(sys.argv[1])))
