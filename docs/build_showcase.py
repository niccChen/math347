"""Render README figures from committed experiment CSVs (no optimizer reruns)."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs' / 'assets'
OUT.mkdir(parents=True, exist_ok=True)
COLORS = {'GD': '#4c78a8', 'L-BFGS': '#8066aa', 'ALS': '#368e83'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#d5dce5','axes.labelcolor':'#637080','xtick.color':'#637080','ytick.color':'#637080','text.color':'#25354b','figure.facecolor':'#fafbfd','axes.facecolor':'#fafbfd','grid.color':'#e7ecf1','grid.alpha':1})
with (ROOT/'results/experiment_1/experiment_1_summary.csv').open() as f: rows=list(csv.DictReader(f))
datasets=['synthetic_dense','synthetic_sparse']; methods=['GD','L-BFGS','ALS']
fig,axes=plt.subplots(1,3,figsize=(14,4.7))
for ax,field,title,ylabel in zip(axes,['time_seconds','iterations','recovery_error'],['Recorded runtime','Optimization steps','Recovery of the clean signal'],['Seconds (log scale)','Iterations','Relative recovery error (%)']):
 for i,method in enumerate(methods):
  values=[float(next(r[field] for r in rows if r['dataset']==d and r['method']==method)) for d in datasets]
  if field=='recovery_error':values=[100*v for v in values]
  ax.bar(np.arange(2)+(i-1)*.24,values,width=.21,color=COLORS[method],label=method,zorder=3)
 ax.set_xticks([0,1],['Dense','Sparse']);ax.set_title(title,loc='left',fontweight='bold',pad=17);ax.set_ylabel(ylabel);ax.grid(axis='y',zorder=0)
 if field=='time_seconds':ax.set_yscale('log');ax.set_ylim(.01,5)
 else:ax.set_ylim(0,max(p.get_height() for p in ax.patches)*1.22)
fig.suptitle('Different optimization paths. Similar recovery quality.',x=.06,y=.99,ha='left',fontsize=18,fontweight='bold')
fig.legend(*axes[0].get_legend_handles_labels(),loc='upper right',bbox_to_anchor=(.96,.99),frameon=False,ncol=3)
fig.text(.06,.015,'Recorded Experiment 1 · rank 5 · λ = 0.01 · timing reflects the archived run, not a portable speed guarantee.',fontsize=9,color='#7c8797')
fig.tight_layout(rect=[.025,.06,.995,.89]);fig.savefig(OUT/'optimizer-comparison.png',dpi=170);plt.close(fig)
with (ROOT/'results/experiment_3/experiment_3_summary.csv').open() as f:rows=list(csv.DictReader(f))
fig,ax=plt.subplots(figsize=(10.5,4.5))
for method in methods:
 r=sorted([x for x in rows if x['dataset']=='synthetic_dense' and x['method']==method],key=lambda x:int(x['rank']))
 ax.plot([int(x['rank']) for x in r],[float(x['reconstruction_error']) for x in r],marker='o',lw=2,color=COLORS[method],label=method)
ax.axvline(5,color='#c1a765',ls='--',lw=1.4,label='Ground-truth rank = 5');ax.set_xlabel('Model rank');ax.set_ylabel('Relative reconstruction error');ax.set_title('Rank selection changes the problem more than the solver.',loc='left',pad=19,fontweight='bold',fontsize=16);ax.grid();ax.legend(frameon=False,ncol=2);fig.tight_layout();fig.savefig(OUT/'rank-sensitivity.png',dpi=170);plt.close(fig)
# Code-native, illustrative factorization graphic for README and portfolio branding.
rects=[]
for x,y,rows,cols,cell,color,label in [(885,182,7,10,24,'#7898d4','M'),(885,428,7,3,17,'#69b6a5','U'),(1021,428,3,10,17,'#b39acf','Vᵀ')]:
 for i in range(rows):
  for j in range(cols):
   opacity=.25+.65*((i*7+j*3)%11)/10
   rects.append(f'<rect x="{x+j*(cell+4)}" y="{y+i*(cell+4)}" width="{cell}" height="{cell}" rx="3" fill="{color}" fill-opacity="{opacity:.2f}"/>')
 label_y = y-15 if label == 'M' else y+rows*(cell+4)+20
 rects.append(f'<text x="{x}" y="{label_y}" fill="#aebad1" font-family="Georgia,serif" font-size="24">{label}</text>')
svg='''<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="800" viewBox="0 0 1600 800" role="img" aria-labelledby="t d"><title id="t">RankLab — Low-Rank Matrix Optimization</title><desc id="d">Three optimizers, four data settings, five experiments. An illustrative matrix decomposition M approximately equals U times V transpose.</desc><defs><linearGradient id="bg" x2="1" y2="1"><stop stop-color="#15223b"/><stop offset="1" stop-color="#24334e"/></linearGradient></defs><rect width="1600" height="800" fill="url(#bg)"/><circle cx="1470" cy="100" r="410" fill="#b8acdd" fill-opacity=".045"/><circle cx="1390" cy="670" r="350" fill="none" stroke="#92bcb0" stroke-opacity=".12"/><path d="M70 70H1530" stroke="#b7c5de" stroke-opacity=".17"/><text x="80" y="127" font-family="Arial,sans-serif" font-size="17" letter-spacing="4" fill="#9bb1cc">NONLINEAR OPTIMIZATION / EMORY UNIVERSITY</text><text x="72" y="298" font-family="Arial,sans-serif" font-weight="700" font-size="126" letter-spacing="-7" fill="#f6f8fd">RankLab<tspan fill="#8ac5b3">.</tspan></text><text x="80" y="362" font-family="Arial,sans-serif" font-size="34" fill="#c4d2e6">Low-Rank Matrix Optimization</text><text x="82" y="419" font-family="Arial,sans-serif" font-size="21" fill="#8fa4bf">Factorize the data. Compare the optimizers.</text><text x="82" y="451" font-family="Arial,sans-serif" font-size="21" fill="#8fa4bf">Understand the role of rank.</text><g font-family="Arial,sans-serif" font-size="17"><rect x="80" y="507" width="104" height="43" rx="21" fill="#4c78a8" fill-opacity=".28" stroke="#658cb7"/><text x="132" y="535" fill="#d9e7f8" text-anchor="middle">GD</text><rect x="198" y="507" width="143" height="43" rx="21" fill="#8066aa" fill-opacity=".28" stroke="#9f87c5"/><text x="269" y="535" fill="#eadff8" text-anchor="middle">L-BFGS</text><rect x="355" y="507" width="104" height="43" rx="21" fill="#368e83" fill-opacity=".28" stroke="#6ab4a7"/><text x="407" y="535" fill="#d0eee5" text-anchor="middle">ALS</text></g><g transform="translate(65 0)">MATRICES<path d="M1024 387v21m-7-7 7 7 7-7" stroke="#9aacca" stroke-width="2" fill="none"/><text x="963" y="475" font-family="Georgia,serif" font-size="27" fill="#b2c2d9">×</text></g><path d="M80 628H1520" stroke="#b7c5de" stroke-opacity=".17"/><g font-family="Arial,sans-serif"><text x="80" y="694" fill="#f1f6fc" font-size="35">3<tspan dx="14" fill="#8fa4bf" font-size="17">OPTIMIZERS</tspan></text><text x="490" y="694" fill="#f1f6fc" font-size="35">4<tspan dx="14" fill="#8fa4bf" font-size="17">DATA SETTINGS</tspan></text><text x="1000" y="694" fill="#f1f6fc" font-size="35">5<tspan dx="14" fill="#8fa4bf" font-size="17">EXPERIMENTAL QUESTIONS</tspan></text><text x="80" y="757" fill="#647c9a" font-size="14" letter-spacing="2">PYTHON · NUMPY · SCIPY · PANDAS · MATPLOTLIB</text></g></svg>'''.replace('MATRICES',''.join(rects))
(OUT/'ranklab-banner.svg').write_text(svg)
print('Rendered the RankLab banner and two figures from committed CSV results.')
