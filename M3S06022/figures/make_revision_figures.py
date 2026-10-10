"""Manuscript figures from the revised cohort and prediction outputs.
Run only after analysis cohort and predictions are finalized. Exports aggregate
source data only; participant-level prediction inputs stay in the restricted
analysis folder. This script does not alter any fitted model or outcome.
"""
from pathlib import Path
import sys, json, argparse
HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent / 'audit_analysis'
sys.path.insert(0, str(ANALYSIS / '_deps'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import LogisticRegression

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'text.color': 'black', 'axes.labelcolor': 'black', 'xtick.color': 'black',
    'ytick.color': 'black', 'axes.titlecolor': 'black',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .7, 'xtick.major.width': .65, 'ytick.major.width': .65,
    'svg.fonttype': 'none', 'pdf.fonttype': 42, 'legend.frameon': False,
    'savefig.facecolor': 'white', 'figure.facecolor': 'white',
})
MODELS = ['M3_continuous_SIC', 'Peak_SOFA', 'Serial_SOFA', 'Expanded_XGB']
LABELS = {'M3_continuous_SIC': 'Continuous SIC (M3)', 'Peak_SOFA': 'Peak SOFA',
          'Serial_SOFA': 'Serial SOFA', 'Expanded_XGB': 'Expanded XGBoost'}
COLORS = {'M3_continuous_SIC': '#A34332', 'Peak_SOFA': '#737373',
          'Serial_SOFA': '#B18A36', 'Expanded_XGB': '#3E7865'}
STYLES = {'M3_continuous_SIC': '-', 'Peak_SOFA': '--', 'Serial_SOFA': '-.', 'Expanded_XGB': ':'}
MARKERS = {'M3_continuous_SIC': 'o', 'Peak_SOFA': 's', 'Serial_SOFA': '^', 'Expanded_XGB': 'D'}
CAPTIONS = {}


def label(ax, s, offset=(-31, 10)):
    ax.annotate(s, xy=(0, 1), xycoords='axes fraction', xytext=offset,
                textcoords='offset points', ha='left', va='bottom',
                fontsize=10, fontweight='bold', annotation_clip=False)


def export(fig, stem, axes, title):
    fig.canvas.draw()
    # Authoring-environment alignment audit omitted from the portable code export.
    fig.savefig(HERE / f'{stem}.pdf', metadata={'Title': title, 'Creator': 'Matplotlib'})
    fig.savefig(HERE / f'{stem}.svg')
    fig.savefig(HERE / f'{stem}.png', dpi=600)
    plt.close(fig)


def flow_box(ax, y, h, text, fill='white', bold=False):
    ax.add_patch(FancyBboxPatch((.035, y), .93, h,
                  boxstyle='round,pad=0.008,rounding_size=0.008',
                  linewidth=.75, edgecolor='#555555', facecolor=fill,
                  transform=ax.transAxes, clip_on=False))
    ax.text(.5, y+h/2, text, ha='center', va='center',
            transform=ax.transAxes, fontsize=8,
            fontweight='bold' if bold else 'normal', linespacing=1.45)


def arrow(ax, ya, yb):
    ax.annotate('', xy=(.5, yb), xytext=(.5, ya), xycoords='axes fraction',
                arrowprops={'arrowstyle': '-|>', 'lw': .7, 'color': '#555555'})


def figure1():
    d = json.loads((ANALYSIS/'cohort_audit.json').read_text(encoding='utf-8'))
    fig, axs = plt.subplots(1, 2, figsize=(6.45, 5.5))
    fig.subplots_adjust(left=.035, right=.985, top=.91, bottom=.035, wspace=.10)
    a, b = d['cohort_mimic'], d['cohort_eicu']
    for ax, key, z, name in zip(axs, ['a','b'], [a,b], ['MIMIC-IV derivation','eICU external application']):
        ax.set_axis_off(); label(ax,key,offset=(0,10)); ax.set_title(name,pad=12)
        source = z.get('source',z.get('text_sepsis'))
        definition = 'Candidate Sepsis-3 admissions' if key=='a' else 'Sepsis text known by 72 h'
        flow_box(ax,.85,.13,f'{definition}\nn = {source:,}',bold=True)
        ex = f"Before 72 h: ICU death {z['early_ICU_death']:,}\nICU discharge alive {z['early_ICU_discharge_alive']:,}"
        # Additional recorded-death/chronology conflicts must be disjoint from early ICU exclusions.
        extra = sum(z.get(k,0) for k in ('ICU_still_present_but_recorded_hospital_death_before72','subsequent_exclusion_hospital_discharge_by72','subsequent_exclusion_dateonly_death_before_landmark_date'))
        if extra: ex += f'\nAdditional chronology exclusions {extra:,}'
        pair=z.get('subsequent_exclusion_infection_pair_not_known_by72',0)
        if pair: ex += f'\nInfection pair not known by 72 h: {pair:,}'
        assert source-z['early_total']-extra-pair==z['landmark_before_W1_filter']
        unknown=z.get('subsequent_exclusion_unknown_hospital_outcome',0)
        assert z['landmark_before_W1_filter']-z['missing_W1_SIC']-unknown==z['included']
        flow_box(ax,.615,.19,ex,fill='#F1F1EF')
        arrow(ax,.847,.81)
        flow_box(ax,.455,.115,f"Eligible at 72-h landmark\nn = {z['landmark_before_W1_filter']:,}")
        arrow(ax,.607,.577)
        filter_text=f"First-window SIC unavailable\nExcluded: n = {z['missing_W1_SIC']:,}"
        if unknown: filter_text+=f'\nThen unknown hospital outcome: {unknown:,}'
        flow_box(ax,.28,.13,filter_text,fill='#F1F1EF')
        arrow(ax,.447,.417)
        flow_box(ax,.08,.15,f"Final landmark cohort\nn = {z['included']:,}\nHospital deaths: {z['hospital_death']:,} ({z['hospital_event_rate_pct']:.1f}%)",fill='#F6ECE8',bold=True)
        arrow(ax,.272,.246)
        ax.text(.5,.035,'Alive and in ICU at 72 h',ha='center',va='center',transform=ax.transAxes,fontsize=8)
    export(fig,'figure1_cohort_flow',{'a':axs[0],'b':axs[1]},'Cohort construction for the 72-hour landmark study')
    (HERE/'figure1_source.json').write_text(json.dumps(d,indent=2),encoding='utf-8')
    CAPTIONS['figure1'] = (
        'Figure 1. Cohort construction for the 72-hour landmark study. '
        '(a) MIMIC-IV derivation cohort identified using Sepsis-3 criteria. '
        '(b) eICU external cohort identified using diagnostic and APACHE text. '
        'Exclusions are applied sequentially. Records with unknown hospital discharge status are excluded after the first-window SIC requirement. MIMIC-IV requires both culture collection and antibiotic prescription start to be known by 72 hours; eICU requires baseline APACHE sepsis text or qualifying diagnosis text documented by 72 hours. Additional chronology exclusions in MIMIC-IV comprise 54 recorded hospital deaths before 72 hours, 33 hospital discharges by 72 hours, and one date-only death before the landmark date; eICU excludes eight hospital discharges by 72 hours. The primary endpoint is hospital death during the index hospitalization after the 72-hour landmark. '
        'The sepsis definitions differ between databases; the external cohort therefore evaluates transport across non-equivalent source populations.')


def bin_calibration(y,p,model,cohort,groups):
    bins = pd.qcut(p,10,duplicates='drop')
    codes=bins.codes
    d=pd.DataFrame({'p':p,'y':y,'bin':bins})
    _,g=np.unique(groups,return_inverse=True)
    ng=g.max()+1; nb=len(bins.categories)
    nn=np.zeros((ng,nb),dtype=np.int64); ee=np.zeros((ng,nb),dtype=np.int64)
    np.add.at(nn,(g,codes),1); np.add.at(ee,(g,codes),y.astype(np.int64))
    rng=np.random.default_rng(42); boot=[]
    for _ in range(1000):
        ix=rng.integers(0,ng,ng)
        n=nn[ix].sum(axis=0); e=ee[ix].sum(axis=0)
        boot.append(np.divide(e,n,out=np.full(nb,np.nan),where=n>0))
    lo,hi=np.nanquantile(np.array(boot),[.025,.975],axis=0)
    rows=[]
    for j,(_,q) in enumerate(d.groupby('bin',observed=True)):
        rows.append({'cohort':cohort,'model':model,'n':len(q),'events':int(q.y.sum()),
                     'mean_predicted':q.p.mean(),'observed':q.y.mean(),
                     'observed_lo':lo[j],'observed_hi':hi[j],
                     'interval':'1000-bootstrap percentile; fixed prediction deciles',
                     'resampling_unit':'patient' if cohort=='MIMIC-IV' else 'patient cluster'})
    return pd.DataFrame(rows)


def flexible_calibration(y,p):
    # Descriptive logistic cubic B-spline, four quantile knots, fixed L2 C=1.
    # Fit only to describe calibration; it does not update prediction outputs.
    p=np.clip(p,1e-6,1-1e-6)
    x=np.log(p/(1-p)).reshape(-1,1)
    spline=SplineTransformer(n_knots=4,degree=3,knots='quantile',include_bias=False)
    xx=spline.fit_transform(x)
    lr=LogisticRegression(C=1,max_iter=2000).fit(xx,y)
    pp=np.linspace(np.quantile(p,.01),np.quantile(p,.99),200)
    xxg=spline.transform(np.log(pp/(1-pp)).reshape(-1,1))
    return pp,lr.predict_proba(xxg)[:,1]


def figure2():
    datasets={
        'MIMIC-IV':pd.read_csv(ANALYSIS/'internal_predictions_RESTRICTED.csv'),
        'eICU':pd.read_csv(ANALYSIS/'external_predictions_RESTRICTED.csv')}
    for name,d in datasets.items():
        assert set(MODELS).issubset(d), f'Models not ready: {name}'
        assert d[MODELS+['y']].notna().all().all(), f'Missing predictions: {name}'
        assert d.y.isin([0,1]).all()
        audit=json.loads((ANALYSIS/'cohort_audit.json').read_text())
        key='cohort_mimic' if name=='MIMIC-IV' else 'cohort_eicu'
        assert len(d)==audit[key]['included'] and d.y.sum()==audit[key]['hospital_death'], 'Predictions and cohort audit differ'
    fig,axs=plt.subplots(2,2,figsize=(6.45,5.7))
    fig.subplots_adjust(left=.105,right=.975,top=.925,bottom=.16,hspace=.45,wspace=.29)
    rocs=[]; cals=[]; smooth=[]
    for col,(cohort,d) in enumerate(datasets.items()):
        y=d.y.to_numpy(); ax=axs[0,col]; ca=axs[1,col]
        ax.plot([0,1],[0,1],color='#B6B6B6',lw=.7,ls='--',zorder=0)
        ca.plot([0,1],[0,1],color='#B6B6B6',lw=.7,ls='--',zorder=0)
        for model in MODELS:
            p=d[model].to_numpy(); fpr,tpr,_=roc_curve(y,p)
            assert np.all(np.diff(fpr)>=0)
            ax.plot(fpr,tpr,color=COLORS[model],ls=STYLES[model],lw=1.4,label=LABELS[model])
            rocs.extend({'cohort':cohort,'model':model,'false_positive_rate':float(f),'true_positive_rate':float(t)} for f,t in zip(fpr,tpr))
            groups=d['stay_id' if cohort=='MIMIC-IV' else 'uniquepid'].to_numpy()
            cal=bin_calibration(y,p,model,cohort,groups); cals.append(cal)
            xs,ys=flexible_calibration(y,p)
            ca.plot(xs,ys,color=COLORS[model],ls=STYLES[model],lw=1.3)
            ca.errorbar(cal.mean_predicted,cal.observed,
                yerr=np.vstack([cal.observed-cal.observed_lo,cal.observed_hi-cal.observed]),
                fmt=MARKERS[model],color=COLORS[model],markersize=3.1,
                markerfacecolor='white',markeredgewidth=.7,elinewidth=.5,
                capsize=1.5,alpha=.9,zorder=3)
            smooth.extend({'cohort':cohort,'model':model,'predicted':float(x),'smoothed_observed':float(z)} for x,z in zip(xs,ys))
        ax.set(xlim=(0,1),ylim=(0,1),xlabel='False-positive rate',ylabel='True-positive rate')
        ca.set(xlim=(0,1),ylim=(0,1),xlabel='Predicted hospital mortality',ylabel='Observed hospital mortality')
        for aa in (ax,ca):
            aa.set_xticks(np.linspace(0,1,6));aa.set_yticks(np.linspace(0,1,6))
            aa.tick_params(length=3,pad=3)
        ax.set_title('Internal cross-validation' if col==0 else 'External application',pad=9)
        ca.set_title(f'{cohort} calibration',pad=9)
        label(ax,'ab'[col]);label(ca,'cd'[col])
    handles,labels=axs[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.54,.028),ncol=2,
               columnspacing=2.5,handlelength=2.5,labelspacing=.75)
    export(fig,'figure2_discrimination_calibration',dict(zip('abcd',[axs[0,0],axs[0,1],axs[1,0],axs[1,1]])),
           'Discrimination and calibration in derivation and external cohorts')
    pd.DataFrame(rocs).to_csv(HERE/'figure2_roc_source.csv',index=False)
    pd.concat(cals).to_csv(HERE/'figure2_calibration_bins_source.csv',index=False)
    pd.DataFrame(smooth).to_csv(HERE/'figure2_calibration_spline_source.csv',index=False)
    n1=len(datasets['MIMIC-IV']);n2=len(datasets['eICU'])
    CAPTIONS['figure2']=(
        'Figure 2. Discrimination and calibration after harmonized scoring and leakage-controlled model development. '
        f'(a) Receiver-operating-characteristic curves from five outer-fold predictions in MIMIC-IV (n = {n1:,}). '
        f'(b) External model application in eICU (n = {n2:,}). '
        '(c, d) Calibration in the corresponding cohorts. Points are observed hospital mortality within deciles of predicted risk; error bars are 95% percentile intervals. '
        'Intervals use 1,000 bootstrap samples of patients in MIMIC-IV and patient clusters in eICU, conditional on the fitted predictions and decile boundaries. Curves are descriptive logistic cubic B-spline smooths with four quantile knots and fixed L2 regularization (C = 1), displayed from the 1st to 99th prediction percentiles. '
        'Smoothing does not modify model predictions. The diagonal denotes perfect calibration in panels c and d and chance discrimination in panels a and b. '
        'All model preprocessing and tuning used derivation data only; external outcomes were used for evaluation. '
        'SIC, sepsis-induced coagulopathy; SOFA, Sequential Organ Failure Assessment. Model-specific performance intervals are reported in the accompanying tables.')


def figure3():
    profiles=pd.read_csv(ANALYSIS/'cluster_profiles.csv')
    boot=pd.read_csv(ANALYSIS/'cluster_profile_bootstrap.csv')
    stability=pd.read_csv(ANALYSIS/'cluster_stability_distribution.csv')
    selection=json.loads((ANALYSIS/'cluster_selection.json').read_text(encoding='utf-8'))
    silhouettes={int(v['k']):float(v['silhouette']) for v in selection}
    rows=[];summary=[]
    fig,axs=plt.subplots(1,2,figsize=(6.45,3.6))
    fig.subplots_adjust(left=.09,right=.985,top=.88,bottom=.35,wspace=.28)
    for ax,k,letter in zip(axs,[2,4],['a','b']):
        q=profiles.loc[(profiles.k==k)&profiles.cohort.eq('MIMIC-IV')].sort_values('stratum')
        assert q.n.sum()==json.loads((ANALYSIS/'cohort_audit.json').read_text())['cohort_mimic']['included']
        colors=['#3E7865','#A34332'] if k==2 else ['#3E7865','#737373','#B18A36','#A34332']
        styles=['-','--','-.',':'];marks=['o','s','^','D']
        for ii,(_,r) in enumerate(q.iterrows()):
            bb=boot.loc[(boot.k==k)&(boot.cluster==r.cluster)]
            assert bb.bootstrap.nunique()>=200, 'Wait for completed 200-bootstrap outputs'
            mean=np.array([r[f'w{i}_SIC_mean'] for i in [1,2,3]])
            lo=bb[['w1','w2','w3']].quantile(.025).to_numpy()
            hi=bb[['w1','w2','w3']].quantile(.975).to_numpy()
            name='C'+r.stratum.split()[-1]
            ax.errorbar([1,2,3],mean,yerr=np.maximum(0,np.vstack([mean-lo,hi-mean])),
                fmt=marks[ii],ls=styles[ii],color=colors[ii],lw=1.3,markersize=4,
                capsize=3,elinewidth=.8,label=f'{name} (n = {int(r.n):,})')
            rows.extend({'k':k,'stratum':name,'n':int(r.n),'window':i,'SIC_mean':float(mean[i-1]),
                         'SIC_mean_lo':float(lo[i-1]),'SIC_mean_hi':float(hi[i-1])} for i in [1,2,3])
        ax.set(xlim=(.85,3.15),ylim=(0,6),ylabel='Mean SIC score',xlabel='Hours after ICU admission')
        ax.set_xticks([1,2,3],['0–24','24–48','48–72']);ax.set_yticks([0,1,2,3,4,5,6])
        ax.set_title('Primary K = 2' if k==2 else 'Exploratory K = 4',pad=9)
        label(ax,letter,offset=(-25,10))
        ax.legend(loc='upper center',bbox_to_anchor=(.5,-.30),ncol=2,columnspacing=.9,
                  handlelength=1.6,handletextpad=.5,fontsize=8)
        ari=stability.loc[stability.k==k,'ARI']
        assert len(ari)>=200
        stats={'k':k,'silhouette':silhouettes[k],'ARI_median':float(ari.median()),
               'ARI_p025':float(ari.quantile(.025)),'ARI_p975':float(ari.quantile(.975)),
               'bootstrap_replicates':len(ari)}
        summary.append(stats)
        ax.text(.5,-.56,f"Silhouette {silhouettes[k]:.3f}; median ARI {ari.median():.3f}",
                transform=ax.transAxes,ha='center',va='center',fontsize=8)
    export(fig,'figure3_cluster_trajectories',{'a':axs[0],'b':axs[1]},'Primary and exploratory SIC trajectory partitions')
    pd.DataFrame(rows).to_csv(HERE/'figure3_trajectory_source.csv',index=False)
    pd.DataFrame(summary).to_csv(HERE/'figure3_stability_summary.csv',index=False)
    total=int(profiles.loc[(profiles.k==2)&profiles.cohort.eq('MIMIC-IV'),'n'].sum())
    CAPTIONS['figure3']=(
        'Figure 3. SIC trajectories in the primary and exploratory derivation-cohort partitions. '
        f'(a) Silhouette-selected K = 2. (b) Exploratory K = 4. Both partitions use the same {total:,} eligible MIMIC-IV patients. '
        'C1–C4 denote neutral stratum identifiers ordered by mean SIC across the three windows, independently within each partition; outcomes were not used for this ordering. '
        'Points show within-stratum mean SIC and error bars show 2.5th–97.5th percentiles from 200 patient bootstrap samples conditional on the reference assignments. '
        'These intervals differ from the refitted, aligned centroid intervals reported in the Appendix. Missing later-window SIC inputs were carried from W1 as specified in Methods. '
        'Adjusted Rand index (ARI) summarizes agreement with reference assignments after refitting preprocessing and clustering in each bootstrap; complete stability distributions and external distance diagnostics are supplied in the Appendix. '
        'SIC, sepsis-induced coagulopathy.')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['flow','prediction','cluster','all'])
    args=ap.parse_args()
    if args.mode in ('flow','all'): figure1()
    if args.mode in ('prediction','all'): figure2()
    if args.mode in ('cluster','all'): figure3()
    path=HERE/'captions.json'
    existing=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    existing.update(CAPTIONS)
    path.write_text(json.dumps(existing,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':main()



