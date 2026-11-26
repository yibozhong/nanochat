"""Render a portable HTML snapshot from the 1B synchronizer's published results."""
import base64
from collections import Counter
from html import escape
from pathlib import Path
import re
from string import Template
from urllib.parse import urlsplit

import mistune


def esc(value):
    return escape(str(value), quote=True)


def ce(value):
    return '—' if value is None else f'{value:.5f}'


def data_url(path, mime):
    return 'data:' + mime + ';base64,' + base64.b64encode(path.read_bytes()).decode('ascii')


class ReportHTML(mistune.HTMLRenderer):
    """Only embed artifacts emitted for this topic in the current snapshot."""
    def __init__(self, folder, rows, topic_links):
        super().__init__(escape=True)
        self.folder = folder
        self.topic_links = topic_links
        self.images = {f"curves-{r['protocol_id']}.png" for r in rows if r['protocol_id'] and r['curve']}
        self.downloads = {f"configs/{r['run_id']}.json": 'application/json' for r in rows}
        if rows:
            self.downloads['curves.csv'] = 'text/csv'

    def image(self, text, url, title=None):
        if url not in self.images:
            return esc(text)
        return f'<img loading="lazy" alt="{esc(text)}" src="{data_url(self.folder/url, "image/png")}">'

    def link(self, text, url, title=None):
        if url in self.downloads:
            href = data_url(self.folder/url, self.downloads[url])
            return f'<a href="{href}" download="{esc(Path(url).name)}">{text}</a>'
        parsed = urlsplit(url)
        if parsed.scheme in ('https','http'):
            return super().link(text,url,title)
        if parsed.scheme or parsed.netloc:
            return text
        anchor = self.topic_links.get((self.folder/parsed.path).resolve())
        return f'<a href="#topic-{esc(anchor)}">{text}</a>' if anchor else text

    def table(self, text):
        return f'<div class="table-wrap"><table>{text}</table></div>\n'


def state_pill(status):
    color = {'运行中':'live','已完成':'done','失败':'failed','主动终止':'stopped','已中断':'stopped',
             '挂起待排查':'stopped','最终 CE 高于基线':'negative'}.get(status, 'neutral')
    return f'<span class="status {color}">{esc(status)}</span>'


def grid(headers, rows):
    head = ''.join(f'<th scope="col">{esc(title)}</th>' for title in headers)
    body = ''.join('<tr>'+''.join(f'<td>{cell}</td>' for cell in row)+'</tr>' for row in rows)
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def empty(message):
    return f'<p class="empty">{esc(message)}</p>'


def render(root, groups, ranked, reviewed, synced_at=None, overview_notes=''):
    topics, reports, boards, active, archive = [], [], [], [], []
    topic_links = {(root/g/'REPORT.md').resolve():g for g in groups}
    titles, findings = {}, {}
    for group, rows in sorted(groups.items()):
        folder = root/group
        doc = (folder/'REPORT.md').read_text()
        heading = re.search(r'^# (.+)$',doc,re.M)
        title = titles[group] = heading[1] if heading else group
        body = doc[:heading.start()] + doc[heading.end():] if heading else doc
        # Hide sync/analysis bookkeeping and render author prose with raw HTML escaped.
        body = re.sub(r'<!--(?:\s*(?:agent:|sync:|reviewed:)).*?-->', '',body,flags=re.S)
        md = mistune.create_markdown(renderer=ReportHTML(folder,rows,topic_links),plugins=['table'])
        matches = re.findall(r'^\*\*实测结论\*\*\s+(.+)$', doc, re.M)
        finding = matches[-1] if matches else ''
        manual = re.search(r'^## (?:研究结论|结论)\s*\n+([^#].*?)(?=\n\s*\n|\Z)', body, re.M|re.S)
        if not finding and manual:
            finding = ' '.join(manual[1].split())
        finding_label = '研究者结论' if not matches else '最新结论' if reviewed.get(group) else '上次复核'
        findings[group] = f'{finding_label} · {finding}' if finding else ''
        summary = ' · '.join(f'{s} {n}' for s,n in sorted(Counter(r['status'] for r in rows).items())) or '待实验'
        note = findings[group] or ('暂无已复核结论' if rows else '尚未开训')
        topic = f'topic-{esc(group)}'
        topics.append(f'<article class="topic"><div class="topic-head"><a href="#{topic}">{esc(title)}</a>'
                      f'<span class="meta">{esc(summary)}</span></div><p>{esc(note)}</p></article>')
        reports.append(f'<article class="report" id="{topic}"><header><span class="meta">{esc(group)}</span>'
                       f'<h3>{esc(title)}</h3></header><div class="prose">{md(body)}</div></article>')

    if overview_notes.strip():
        md = mistune.create_markdown(renderer=ReportHTML(root, [], topic_links), plugins=['table'])
        topics.append('<div class="prose">'+md(overview_notes)+'</div>')

    worse = set()
    for cohort, rows in sorted(ranked.items()):
        first = rows[0]
        spec = first['spec']
        context = f"{spec.get('architecture')} · {spec.get('depth')} × {spec.get('width')} · context {spec.get('context')} · seed {first['seed']}"
        entries = []
        for i,r in enumerate(rows,1):
            if r['delta_ce'] > 0:
                worse.add(r['run_id'])
            entries.append([str(i),esc(r['name']),f'<a href="#topic-{esc(r["group"])}">{esc(titles[r["group"]])}</a>',
                            ce(r['final_ce']),ce(r['delta_ce']),esc(r['baseline_run'])])
        boards.append(f'<h3>{esc(context)}</h3><p class="meta">{first["target_tokens"]:,} tokens · 协议 {esc(cohort)}</p>'
                      +grid(['排名','配置','研究问题','最终 CE','ΔCE','基线 ID'],entries))

    interrupted = {'失败','主动终止','已中断'}
    for group, rows in sorted(groups.items()):
        for r in rows:
            title = f'<a href="#topic-{esc(group)}">{esc(titles[group])}</a>'
            name = f'<strong>{esc(r["name"])}</strong><small>{title}</small>'
            status = r['status']
            if status != '已完成' and status not in interrupted:
                n, total = r['trained_tokens'],r['target_tokens']
                if n is not None and total:
                    progress = (f'<strong>{n/total:.1%}</strong><small class="num">{n:,} / {total:,}</small>'
                                f'<progress value="{min(n,total)}" max="{total}" aria-label="训练进度"></progress>')
                elif n is not None:
                    progress = f'{n:,} tokens · 预算未记录'
                else:
                    progress = f'等待训练记录 · 目标 {total:,} tokens' if total else '暂无进度记录'
                active.append([name,state_pill(status),progress,ce(r['validation_ce']),esc(r['gpu'] or '—')])
            elif group == 'legacy-chord-1b' or status in interrupted or r['run_id'] in worse:
                outcome = ('Legacy · ' + status if group == 'legacy-chord-1b' else
                           '最终 CE 高于基线' if status == '已完成' else status)
                notes = []
                if r.get('stop_reason'):
                    notes.append('停止记录 · '+esc(r['stop_reason']))
                if findings[group]:
                    notes.append('专题 · '+esc(findings[group]))
                else:
                    notes.append('专题结论待复核')
                observed = ce(r['validation_ce'])
                if r['validation_tokens'] is not None:
                    observed += f'<small>{r["validation_tokens"]:,} tokens</small>'
                archive.append([name,state_pill(outcome),observed,ce(r['delta_ce']),'<br>'.join(notes)])

    flat = [r for rows in groups.values() for r in rows]
    updated = f'最近同步 <time>{esc(synced_at)}</time>' if synced_at else ''
    return Template(Path(__file__).with_name('research_dashboard.html').read_text()).substitute(
        updated=updated, topic_count=len(groups), run_count=len(flat),
        running_count=sum(r['status']=='运行中' for r in flat),
        overview=''.join(topics) or empty('尚未登记研究方向。'),
        leaderboard=''.join(boards) or empty('暂无同时完成预算的配置与配对基线。'),
        experiments=''.join(reports) or empty('尚无专题报告。'),
        running=grid(['配置 / 研究问题','状态','训练 tokens','最近验证 CE','GPU'],active) if active else empty('当前没有运行中或等待调度的实验。'),
        archive=grid(['配置 / 研究问题','结果','最近验证 CE','ΔCE','记录与结论'],archive) if archive else empty('暂无失败、终止、中断或已确认最终 CE 高于基线的实验。'))
