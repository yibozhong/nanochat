"""Standalone numerical comparison and exportable diagnostic figures."""
import csv
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.connection_depth_diagnostic import summarized


def main(root):
    root = Path(root)
    result = json.loads((root/'results.json').read_text())
    models = result['models']
    colors = dict(vanilla='#3975b9', native='#df8a23', mhc='#278b66')
    labels = dict(vanilla='Vanilla', native='Native nanochat', mhc='Vanilla + mHC')
    def lines(ax, key, subset='layers', label=None):
        for name,m in models.items():
            rows=[r for r in m[subset] if key in r]
            x=[r['normalized_depth'] for r in rows]
            y=[r[key]['mean'] for r in rows]
            ax.plot(x,y,'o-',label=labels[name],color=colors[name],markersize=4)
            ax.fill_between(x,[r[key]['ci95'][0] for r in rows],[r[key]['ci95'][1] for r in rows],color=colors[name],alpha=.12)
        ax.set_xlabel('Normalized depth (layer / 8)'); ax.set_ylabel(label or key)
        ax.grid(alpha=.2)
    fig,axs=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    for ax,key,label in zip(axs.flat,
        ['total_ratio','compute_ratio','routing_ratio','delta_ce','adjacent_total_cosine','adjacent_compute_cosine'],
        ['Total update / residual','Computational increment / residual','Routing increment / residual',
         'Validation CE increase after branch skip','Adjacent total-update cosine','Adjacent compute-update cosine']):
        lines(ax,key,label=label)
    axs[1,0].axhline(0,color='black',lw=.7); axs[0,0].legend()
    fig.suptitle('Depth utilization: 8-layer models, 10B training tokens, identical 2K dev contexts')
    for ext in ('png','pdf'):fig.savefig(root/('comparison.'+ext),dpi=180)
    plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(13,3.8),constrained_layout=True)
    for ax,key,label in zip(axs,['hidden_norm','hidden_variance','hidden_cosine'],['Residual L2 norm','Residual coordinate variance','Adjacent hidden-state cosine']):
        lines(ax,key,label=label)
    axs[0].legend()
    for ext in ('png','pdf'):fig.savefig(root/('residual-geometry.'+ext),dpi=180)
    plt.close(fig)
    fields=['model','layer','normalized_depth','baseline_ce','skip_ce','delta_ce','delta_ce_ci_low','delta_ce_ci_high',
            'total_ratio','compute_ratio','routing_ratio','total_frobenius_ratio','compute_frobenius_ratio',
            'adjacent_total_cosine','adjacent_compute_cosine','hidden_norm','hidden_variance']
    with (root/'per-layer.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for name,m in models.items():
            for r in m['layers']:
                row=dict(model=name,layer=r['layer'],normalized_depth=r['normalized_depth'],baseline_ce=m['baseline_ce']['mean'],
                         delta_ce_ci_low=r['delta_ce']['ci95'][0],delta_ce_ci_high=r['delta_ce']['ci95'][1])
                for key in fields:
                    if key not in row and key in r:row[key]=r[key]['mean'] if isinstance(r[key],dict) else r[key]
                writer.writerow(row)
    arrays={name:np.load(root/(name+'-per-sequence.npz')) for name in models}
    paired={}
    for other in ('mhc','native'):
        a,b=arrays[other]['delta_ce'],arrays['vanilla']['delta_ce']
        paired[other+'_minus_vanilla']=dict(
            baseline_ce=summarized(arrays[other]['baseline_ce']-arrays['vanilla']['baseline_ce']),
            layer_delta_ce=[summarized(a[:,i]-b[:,i]) for i in range(8)],
            late_mean_delta_ce=summarized((a[:,4:]-b[:,4:]).mean(1)),
            late_except_final_delta_ce=summarized((a[:,4:7]-b[:,4:7]).mean(1)),
            depth_profile_interaction=summarized((a[:,4:]-b[:,4:]).mean(1)-(a[:,:4]-b[:,:4]).mean(1)))
    (root/'paired-comparison.json').write_text(json.dumps(paired,indent=2)+'\n')
    print(json.dumps(paired,indent=2))


if __name__=='__main__':main(sys.argv[1])
