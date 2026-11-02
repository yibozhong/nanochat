"""Same-device 2K/4K/8K/16K comparison with two reversed-order repetitions."""
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

LENGTHS = (2048, 4096, 8192, 16384)
VARIANTS = ('baseline', 'mhc')


def main():
    root = Path('results/context'); root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path('../data/manifest.json').read_text())
    assert manifest['sequence_length'] == 2048
    digest = hashlib.sha256()
    with Path('../data/train.bin').open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024), b''): digest.update(chunk)
    assert digest.hexdigest() == manifest['files']['train']['sha256']
    probes = []; trials = []; selected = {}; unsupported = {}
    def run(length, variant, micro, repetition=0, probe=False):
        key = f'{length}-{variant}-mb{micro}-r{repetition}'+('-probe' if probe else '')
        output = root/(key+'.json')
        cmd = [sys.executable, '-m', 'scripts.connection_context_profile', '--variant', variant,
            '--sequence-length', str(length), '--microbatch', str(micro),
            '--repetition', str(repetition), '--output', str(output)]
        if probe: cmd.append('--probe')
        print('START '+key, flush=True)
        with (root/(key+'.log')).open('w') as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = json.loads(output.read_text())
        assert result['logging_backend']=='local' and result['status'] in ('fits','over_budget','oom')
        print('END '+key+' '+result['status'], flush=True)
        return result
    # Use the desired no-accumulation shape if it meets the fixed 22GB budget.
    # A failed capacity attempt never poisons the process for the next model.
    for length in LENGTHS:
        micro = 32768//length
        while micro:
            pair = [run(length, v, micro, probe=True) for v in VARIANTS]
            probes.extend(pair)
            if all(p['status']=='fits' for p in pair):
                selected[length] = micro; break
            micro //= 2
        if length not in selected:
            unsupported[length] = 'No common microbatch choice meets the 22GB budget; no checkpoint fallback was enabled.'
    for repetition in (0, 1):
        lengths = LENGTHS if repetition==0 else tuple(reversed(LENGTHS))
        variants = VARIANTS if repetition==0 else tuple(reversed(VARIANTS))
        for length in lengths:
            if length not in selected: continue
            for variant in variants:
                trial = run(length, variant, selected[length], repetition)
                assert trial['status']=='fits' and trial['causality_passed'], trial
                assert len(trial['step_seconds'])==40 and trial['optimizer_updates']==60
                trials.append(trial)
    assert len({x['gpu_uuid'] for x in probes+trials})==1
    rows = []
    for length in selected:
        for variant in VARIANTS:
            pair = [x for x in trials if x['sequence_length']==length and x['variant']==variant]
            assert len(pair)==2 and len({x['input_sha256'] for x in pair})==1
            times = [v for t in pair for v in t['step_seconds']]
            rows.append(dict(sequence_length=length, variant=variant, microbatch=selected[length],
                grad_accum=(32768//length)//selected[length], mean_ms=1000*statistics.mean(times),
                median_ms=1000*statistics.median(times), tokens_per_second=32768/statistics.mean(times),
                peak_reserved_gib=max(x['peak_reserved_bytes'] for x in pair)/1024**3,
                peak_allocated_gib=max(x['peak_allocated_bytes'] for x in pair)/1024**3,
                repetition_mean_ms=[x['mean_seconds']*1000 for x in pair],
                estimated_10b_training_hours=305176*statistics.mean(times)/3600))
    for row in rows:
        baseline = next((x for x in rows if x['sequence_length']==2048 and x['variant']==row['variant']), None)
        row['relative_to_same_variant_2k'] = row['mean_ms']/baseline['mean_ms'] if baseline else None
    result = dict(complete=True, source=json.loads(Path('source.json').read_text()), logging_backend='local',
        gpu_uuid=(probes+trials)[0]['gpu_uuid'], gpu=(probes+trials)[0]['gpu'],
        batch_tokens=32768, lengths=list(LENGTHS), variants=list(VARIANTS),
        selected_microbatches=selected, unsupported=unsupported, summary=rows,
        probes=probes, trials=trials, original_train_sha256=digest.hexdigest(),
        scope='Full-attention vanilla vs NVIDIA mHC only; no native SSSL or no-Sinkhorn measurement.',
        limitations='Resegmented existing packed tokens; no quality/long-document claim. Time excludes setup/compile/evaluation/checkpoint transfer. Fixed 22GB allocation budget is not a physical 22GB GPU test.')
    Path('results/context-results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == '__main__':
    main()
