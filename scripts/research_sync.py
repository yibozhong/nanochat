# /// script
# requires-python = ">=3.11"
# dependencies = ["wandb==0.26.1", "matplotlib==3.10.8", "mistune==3.2.0"]
# ///
"""Synchronize the 1B project's full validation curves and Chinese research reports."""
import argparse
import csv
import fcntl
import hashlib
import io
import json
import math
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from nanochat.experiment_logging import ENTITY, PROJECT
from scripts.research_dashboard import render as render_dashboard

COLUMNS = ('group name run_id status wandb_state baseline_run protocol_id target_tokens trained_tokens '
           'validation_tokens validation_ce final_ce best_ce delta_ce comparison parameters '
           'estimated_training_flops flops_scope tokens_per_second peak_allocated_bytes training_seconds '
           'gpu seed dataset_revision git_commit url').split()
BEGIN = '<!-- sync:begin -->'
END = '<!-- sync:end -->'


def atomic(path, content):
    content = content.encode() if isinstance(content, str) else content
    if path.exists() and path.read_bytes() == content:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        tmp.write(content)
    os.chmod(tmp.name, 0o644)
    os.replace(tmp.name, path)


def table(columns, rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore', lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def update_block(path, initial, text):
    doc = path.read_text() if path.exists() else initial
    if doc.count(BEGIN) != 1 or doc.count(END) != 1:
        raise ValueError(f'{path}: automatic section markers missing or duplicated')
    left, rest = doc.split(BEGIN)
    _, right = rest.split(END)
    atomic(path, left + BEGIN + '\n\n' + text + '\n\n' + END + right)


def scheduler_states(jobs):
    # Read only directories created by the new launcher. Do not touch legacy runs.
    local = {}
    for path in sorted(jobs.glob(f'{PROJECT}-*/state.json')):
        state = json.loads(path.read_text())
        source = state.get('source', {})
        runs = source.get('runs', [source])
        stopped = path.with_name('STOPPED.json')
        stop = json.loads(stopped.read_text()) if stopped.is_file() else {}
        active = None
        if 'runs' in source:
            log = path.parent/'logs'/f"train.{state['job']}.out"
            if log.is_file():
                with log.open(errors='replace') as f:
                    for line in f:
                        if line.startswith('RESIDUAL_PHASE '):
                            parts = line.split()
                            if len(parts) == 4:
                                active = parts[2]
        for s in runs:
            if s.get('wandb_project') != PROJECT or not s.get('formal_training'):
                continue
            rid = s['run_id']
            explicit_stop = stop.get('run_id') == rid or rid in stop.get('run_ids', [])
            local[rid] = dict(source=s, job=state['job'], stopped=explicit_stop,
                              suite_wait='runs' in source and rid != active,
                              stop_reason=stop.get('reason') if explicit_stop else None)
    if not local or not shutil.which('condor_q'):
        return local
    by_job = defaultdict(list)
    for value in local.values():
        by_job[value['job']].append(value)
    fields = 'ClusterId,ProcId,JobStatus,ExitCode,ExitBySignal'
    def read(command, args):
        output = subprocess.check_output([command, *args, '-attributes', fields, '-json'], text=True, timeout=30)
        for ad in json.loads(output or '[]'):
            job = f"{ad['ClusterId']}.{ad['ProcId']}"
            if job in by_job:
                for entry in by_job[job]:
                    entry['ad'] = ad
    read('condor_q', list(by_job))
    missing = [j for j, v in by_job.items() if 'ad' not in v[0]]
    if missing:
        # Stop traversing the schedd's history once all requested job records are found.
        ids = [tuple(map(int, job.split('.'))) for job in missing]
        constraint = ' || '.join(f'(ClusterId == {cluster} && ProcId == {proc})' for cluster, proc in ids)
        read('condor_history', ['-constraint', constraint, '-limit', str(len(ids))])
    return local


def history(run, cache, refresh):
    # Immutable completed histories are cheap to reuse. Refresh re-reads every event.
    summary = run.summary
    stamp = [summary.get('_step'), summary.get('training_tokens'), summary.get('final_dev_ce')]
    path = cache / (run.id + '.json')
    terminal = summary.get('complete') is True and all(v is not None for v in stamp)
    if terminal and not refresh and path.exists():
        previous = json.loads(path.read_text())
        if previous['stamp'] == stamp:
            return dict(previous['points'])
    points = {}
    for row in run.scan_history(keys=['tokens','dev/ce'], page_size=1000):
        token, ce = row.get('tokens'), row.get('dev/ce')
        if isinstance(token, (float, int)) and token >= 0 and int(token) == token and isinstance(ce, (int,float)) and math.isfinite(ce):
            points[int(token)] = ce
    if terminal and points.get(stamp[1]) is not None and math.isclose(points[stamp[1]],stamp[2],abs_tol=2e-6):
        atomic(path, json.dumps(dict(stamp=stamp, points=sorted(points.items()))))
    return points


def fetch(jobs, cache, refresh=False):
    import wandb
    local = scheduler_states(jobs)
    records = []
    for run in wandb.Api(timeout=90).runs(f'{ENTITY}/{PROJECT}', per_page=100):
        points = history(run, cache, refresh)
        records.append(make_record(run.id, run.name, run.group, run.state, run.config, run.summary,
                                   points, local.pop(run.id, {})))
    for rid, job in local.items():
        s = job['source']
        records.append(make_record(rid, s['run_name'], s['group'], '', {'source':s}, {}, {}, job))
    return sorted(records, key=lambda r: (r['group'], r['name'], r['run_id']))


def make_record(rid, name, group, wb, cfg, summary, curve, job):
    s = cfg.get('source', {})
    # Incomplete external submissions still belong in the inventory.
    group = group if isinstance(group,str) and re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*',group) else 'unassigned'
    name = ' '.join((name or rid).split())
    baseline = s.get('baseline_run') or ''
    p = cfg.get('protocol', {})
    spec = cfg.get('comparison_protocol')
    # Fingerprint of protocol metadata only. No dataset or checkpoint is opened.
    cohort = hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',',':')).encode()).hexdigest()[:16] if spec else ''
    batch, steps = p.get('batch_tokens',s.get('batch_tokens')), p.get('steps',s.get('steps'))
    target = steps*batch if steps and batch else None
    end = max(curve, default=None)
    final = summary.get('final_dev_ce')
    complete = (summary.get('complete') is True and target is not None and
                summary.get('training_tokens') == target and final is not None and math.isfinite(final))
    if complete and (target not in curve or not math.isclose(final, curve[target], abs_tol=2e-6)):
        raise ValueError(f'{rid}: final evaluation is missing or inconsistent with the W&B history')
    ad = job.get('ad', {})
    state = ad.get('JobStatus')
    if complete:
        status = '已完成'
    elif job.get('stopped') or wb == 'killed' or state == 3:
        status = '主动终止' if job.get('stopped') else '已中断'
    elif job.get('suite_wait') and not wb and state in (1, 2, 5):
        status = '挂起待排查' if state == 5 else '排队'
    elif state == 5:
        status = '失败' if ad.get('ExitBySignal') or ad.get('ExitCode') not in (None,0,85) else '挂起待排查'
    elif state == 1:
        status = '等待恢复' if curve else '排队'
    elif state == 2 or wb == 'running':
        loss = summary.get('train/ce')
        has_update = (summary.get('tokens') or 0) > 0 and isinstance(loss,(int,float)) and math.isfinite(loss)
        status = '运行中' if has_update else '准备中'
    elif wb in ('crashed','failed') or ad.get('ExitBySignal') or (state == 4 and ad.get('ExitCode') not in (0,85)):
        status = '失败'
    else:
        status = '等待同步' if not wb or state == 4 and ad.get('ExitCode') == 0 else '已中断'
    trained = summary.get('training_tokens') if complete else summary.get('tokens', end)
    seconds = summary.get('training_seconds') if complete else summary.get('perf/training_seconds')
    speed = trained/seconds if trained is not None and seconds and seconds > 0 else None
    # At evaluation-only rows W&B's global token counter may be ahead of the last perf sample.
    if not complete:
        speed = summary.get('perf/mean_tokens_per_second')
    return dict(group=group,name=name,run_id=rid,status=status,wandb_state=wb,baseline_run=baseline,
                stop_reason=job.get('stop_reason'),
                protocol_id=cohort,target_tokens=target,trained_tokens=trained,validation_tokens=end,
                validation_ce=curve.get(end),final_ce=final if complete else None,best_ce=min(curve.values(),default=None),
                comparison='',delta_ce=None,parameters=cfg.get('parameters'),gpu=cfg.get('gpu'),
                estimated_training_flops=summary.get('estimated_training_flops',summary.get('perf/estimated_training_flops')),
                flops_scope=cfg.get('flops_scope'),tokens_per_second=speed,
                peak_allocated_bytes=summary.get('peak_allocated_bytes',summary.get('perf/peak_allocated_bytes')),
                training_seconds=seconds,seed=p.get('seed',s.get('seed')),dataset_revision=s.get('dataset_revision') or
                cfg.get('data_manifest',{}).get('dataset_revision'),git_commit=s.get('commit'),
                url=f'https://wandb.ai/{ENTITY}/{PROJECT}/runs/{rid}',curve=curve,spec=spec,config=cfg)


def compare(records):
    by_id = {r['run_id']:r for r in records}
    for r in records:
        b = by_id.get(r['baseline_run']) if r['baseline_run'] != 'self' else r
        if not b:
            reason = '等待基线'
        elif not r['spec'] or r['spec'] != b['spec']:
            reason = '协议不匹配或未知'
        elif r['validation_tokens'] not in b['curve']:
            reason = '无同 token 验证点'
        else:
            r['delta_ce'] = r['validation_ce'] - b['curve'][r['validation_tokens']]
            reason = '同协议同 tokens'
        r['comparison'] = reason


def fmt(v):
    return '—' if v is None else f'{v:.5f}'


def label(text):
    return text.replace('|', r'\|').replace('\n',' ')


def plot(path, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, (ax, delta) = plt.subplots(2, 1, sharex=True, figsize=(8,6.5), layout='constrained')
    by_id = {r['run_id']:r for r in rows}
    for r in rows:
        if r['curve']:
            xs = sorted(r['curve'])
            ax.plot([x/1e6 for x in xs], [r['curve'][x] for x in xs],
                    label=f"{r['name']} ({r['run_id']})", marker='.' if len(xs)<3 else None)
            baseline = by_id.get(r['baseline_run'])
            if baseline:
                common = sorted(r['curve'].keys() & baseline['curve'].keys())
                delta.plot([t/1e6 for t in common], [r['curve'][t]-baseline['curve'][t] for t in common],
                           color=ax.lines[-1].get_color(), marker='.' if len(common)<3 else None)
    ax.set_ylabel('Validation CE')
    delta.axhline(0, color='gray', linewidth=.8)
    delta.set(xlabel='Training tokens (millions)', ylabel='CE minus paired baseline')
    delta.grid(alpha=.2)
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    out = io.BytesIO()
    fig.savefig(out, format='png', dpi=130)
    plt.close(fig)
    atomic(path, out.getvalue())


REPORT = '''# {group}

## 假设与实验设计

待研究者填写研究问题、唯一改动和预期判据。

## 实验事实

<!-- sync:begin -->
<!-- sync:end -->

## 分析、结论与待验证问题

待研究者结合完整曲线填写，分别标注事实、解释和待验证假设。
'''
OVERVIEW = '''# Nanochat 1B 研究总览

正式研究使用 [nanochat-1b](https://wandb.ai/yibozhong657-none/nanochat-1b)。默认从零、seed 42，使用同一 packed 数据，30,518 步 × 32,768 tokens。

[每日看板](index.html) · [实验总表](RESULTS.csv) · [同协议排名](LEADERBOARD.md) · [实验规范](../AGENTS.md)

## 研究方向与状态

<!-- sync:begin -->
<!-- sync:end -->

## 核心发现与待验证问题

上方按方向汇总已复核的结论和待验证问题。首次实验前可打开对应 REPORT 查看假设、对照和判据。
'''


def review_ended(folder, stamp, rows):
    # Only the report writer publishes output. Codex reads the evidence in a read-only session.
    with tempfile.TemporaryDirectory() as work:
        out = Path(work)/'answer.json'
        schema = Path(work)/'schema.json'
        fields = ('finding','interpretation','question')
        schema.write_text(json.dumps(dict(type='object',additionalProperties=False,required=fields,
            properties={key:{'type':'string'} for key in fields})))
        facts = [{k:r[k] for k in COLUMNS if k!='url'} for r in rows]
        prompt = (f'用中文分析 nanochat-1b 研究问题 {folder.name}。仅阅读当前问题的 REPORT.md、'
                  'curves.csv 和 configs/，结合下列客观数据。不得操作集群或外部服务。'
                  'finding 写最重要的实测结果及对照、tokens 和 CE；interpretation 解读完整曲线、成本和局限，'
                  '明确标出推测；question 提出一个尚待验证的决定性问题。失败或提前停止也需要分析。'
                  '缺少可比对照时直接指出。每项一到三句，不虚构证据。\n'
                  + json.dumps(facts,ensure_ascii=False))
        subprocess.run(['codex','exec','--ephemeral','--sandbox','read-only','--cd',str(folder),
                        '--output-schema',str(schema),'-o',str(out),'-'],input=prompt,
                       text=True,check=True,timeout=300,stdout=subprocess.DEVNULL)
        review=json.loads(out.read_text())
        if any(not isinstance(review.get(k),str) or not review[k].strip() for k in fields):
            raise ValueError('Incomplete research review')
        cleaned = [' '.join(review[k].split()).replace('<!--','&lt;!--') for k in fields]
    report = folder/'REPORT.md'
    text = report.read_text()
    start, end = '<!-- agent:begin -->', '<!-- agent:end -->'
    if start not in text:
        text = text.rstrip() + '\n\n' + start + '\n' + end + '\n'
    a, block = text.split(start)
    _, b = block.split(end)
    body = f'<!-- reviewed: {stamp} -->\n\n**实测结论** {cleaned[0]}\n\n**分析** {cleaned[1]}\n\n**待验证** {cleaned[2]}'
    atomic(report, a + start + '\n\n' + body + '\n\n' + end + b)


def include_archived_runs(records, root):
    """Restore frozen 1B results even when the original W&B runs are unavailable."""
    archive = root/'legacy-chord-1b'
    index = archive/'runs.json'
    if not index.exists():
        return records
    facts = json.loads(index.read_text())
    rows = [r for r in records if r['run_id'] not in facts]
    for rid, fact in facts.items():
        cfg = json.loads((archive/'configs'/f'{rid}.json').read_text())
        source = cfg['source']
        assert source['run_id'] == rid and source['wandb_project'] == PROJECT
        job = dict(stopped='stop_reason' in fact, stop_reason=fact.get('stop_reason'))
        row = make_record(rid, source['run_name'], archive.name, fact.get('wandb_state', ''), cfg,
                          fact['summary'], dict(fact['points']), job)
        row['url'] = f'{archive.name}/configs/{rid}.json'
        rows.append(row)
    return sorted(rows, key=lambda r: (r['group'], r['name'], r['run_id']))


def publish(records, root, reviewer=None, *, synced_at=None):
    records = include_archived_runs(records, root)
    # A research question can be re-paired after launch. Keep the training manifest intact.
    pairings = root/'analysis_baselines.json'
    if pairings.is_file():
        baselines = json.loads(pairings.read_text())
        for row in records:
            if row['run_id'] in baselines:
                row['baseline_run'] = baselines[row['run_id']]
    compare(records)
    by_id = {r['run_id']:r for r in records}
    groups = defaultdict(list)
    ranked = defaultdict(list)
    for r in records:
        groups[r['group']].append(r)
        baseline = r if r['baseline_run']=='self' else by_id.get(r['baseline_run'])
        if (r['group'] != 'legacy-chord-1b' and r['status']=='已完成' and r['delta_ce'] is not None
                and baseline and baseline['status']=='已完成'):
            ranked[r['protocol_id']].append(r)
    board = ['# 全局 leaderboard', '', '按协议分榜。配置和配对基线都完成全部训练后，按最终 validation CE 升序排列。']
    for cohort, rows in sorted(ranked.items()):
        board.extend(['', f'## {cohort} · {rows[0]["target_tokens"]:,} tokens','',
                      f"{rows[0]['spec'].get('architecture')} · {rows[0]['spec'].get('depth')} 层 × {rows[0]['spec'].get('width')} · context {rows[0]['spec'].get('context')} · seed {rows[0]['seed']}", '',
                      '| 配置 | 研究问题 | 最终 CE | ΔCE | 基线 |','|---|---|---:|---:|---|'])
        rows.sort(key=lambda r:(r['final_ce'],r['run_id']))
        for r in rows:
            board.append(f"| {label(r['name'])} ({r['run_id']}) | [{r['group']}]({r['group']}/REPORT.md) | {fmt(r['final_ce'])} | {fmt(r['delta_ce'])} | {r['baseline_run']} |")
    if not ranked:
        board.extend(['', '暂无符合条件的已完成实验。'])
    overview = []
    reviewed_groups = {}
    # Keep planned and manually authored topics visible even before their first run.
    for p in root.glob('*/REPORT.md'):
        if BEGIN in p.read_text():
            groups.setdefault(p.parent.name, [])
    for group, rows in sorted(groups.items()):
        folder = root/group
        counts = Counter(r['status'] for r in rows)
        overview.append(f"- [{group}]({group}/REPORT.md) · " + ('，'.join(f'{k} {n}' for k,n in sorted(counts.items())) or '待实验'))
        lines = ['| 配置 | 状态 | 验证 tokens | 最近 CE | ΔCE | 基线 |','|---|---|---:|---:|---:|---|']
        for r in rows:
            rid = r['run_id']
            lines.append(f"| [{label(r['name'])}](configs/{rid}.json) | {r['status']} | {r['validation_tokens'] if r['validation_tokens'] is not None else '—'} | {fmt(r['validation_ce'])} | {fmt(r['delta_ce'])} | {r['baseline_run']} |")
            atomic(folder/'configs'/f'{rid}.json', json.dumps(r['config'], ensure_ascii=False, sort_keys=True, indent=2)+'\n')
        stamp = None
        reviewed = False
        if rows and all(r['status'] in ('已完成','失败','主动终止','已中断') for r in rows):
            stamp = ','.join(f"{r['run_id']}@{r['status']}@{r['validation_tokens']}@{fmt(r['validation_ce'])}@{fmt(r['delta_ce'])}" for r in rows)
            previous = folder/'REPORT.md'
            reviewed = previous.exists() and f'<!-- reviewed: {stamp} -->' in previous.read_text()
        if not rows:
            lines = ['暂无正式实验。']
        curves = []
        for cohort in sorted({r['protocol_id'] for r in rows if r['curve']}):
            subset = {r['run_id']:r for r in rows if r['protocol_id']==cohort and r['curve']}
            for r in list(subset.values()):
                b = by_id.get(r['baseline_run'])
                if b and b['spec'] == r['spec'] and b['curve']:
                    subset[b['run_id']] = b
            if not cohort:
                continue
            series = sorted(subset.values(),key=lambda r:(r['name'],r['run_id']))
            image = f'curves-{cohort}.png'
            plot(folder/image, series)
            lines.extend(['',f'![{cohort} validation CE]({image})'])
        for r in rows:
            curves.extend(dict(run_id=r['run_id'],tokens=t,validation_ce=ce)
                          for t,ce in sorted(r['curve'].items()))
        if rows:
            atomic(folder/'curves.csv', table(['run_id','tokens','validation_ce'],curves))
            lines.extend(['','[完整验证曲线数据](curves.csv)'])
        update_block(folder/'REPORT.md', REPORT.format(group=group), '\n'.join(lines))
        if stamp:
            if not reviewed and reviewer:
                reviewed = reviewer(folder, stamp, rows)
            if reviewed:
                report = (folder/'REPORT.md').read_text()
                findings = [line for line in report.splitlines() if line.startswith(('**实测结论**', '**待验证**'))]
                reviewed_groups[group] = True
                overview[-1] += ' · 已复核' + ''.join('\n  - '+line for line in findings)
            else:
                overview[-1] += ' · 待 Agent 复核结论'
    atomic(root/'RESULTS.csv', table(COLUMNS, records))
    atomic(root/'LEADERBOARD.md', '\n'.join(board)+'\n')
    update_block(root/'OVERVIEW.md', OVERVIEW, '\n'.join(overview) or '暂无登记的研究方向。')
    notes = (root/'OVERVIEW.md').read_text().partition(END)[2].strip()
    if notes == OVERVIEW.partition(END)[2].strip():
        notes = ''
    atomic(root/'index.html', render_dashboard(root, groups, ranked, reviewed_groups, synced_at, notes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs-dir', type=Path, default=Path.home()/'nanochat-experiments')
    parser.add_argument('--refresh', action='store_true', help='Bypass completed-run curve cache')
    parser.add_argument('--review-ended', action='store_true', help='Review newly closed groups using local Codex')
    args = parser.parse_args()
    # Serialize API snapshot + publication across cron and manual refreshes.
    lock = Path.home()/'.cache/nanochat-research/sync.lock'
    lock.parent.mkdir(exist_ok=True,parents=True)
    with lock.open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        records = fetch(args.jobs_dir, lock.parent/PROJECT, args.refresh)
        def reviewer(folder, stamp, rows):
            retry = lock.parent/(folder.name+'.retry')
            if retry.exists():
                old = json.loads(retry.read_text())
                if old['stamp'] == stamp and time.time() < old['after']:
                    return False
            try:
                review_ended(folder,stamp,rows)
                return True
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                print(f'{folder.name}: review pending ({error})',file=sys.stderr)
                atomic(retry,json.dumps(dict(stamp=stamp,after=time.time()+6*3600)))
                return False
        publish(records, REPO/'research', reviewer if args.review_ended else None,
                synced_at=datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z'))
    print(f'{ENTITY}/{PROJECT}: {len(records)} runs')


if __name__ == '__main__':
    main()
