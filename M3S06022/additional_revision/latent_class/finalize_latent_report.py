"""Combine aggregate-only lcmm reporting after both planned fit runs finish."""
from pathlib import Path
import json
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'audit_analysis'/'_deps'))
import pandas as pd
import numpy as np

def clean(o):
    if isinstance(o,dict): return {str(k):clean(v) for k,v in o.items()}
    if isinstance(o,list): return [clean(v) for v in o]
    if isinstance(o,(np.integer,np.floating)): o=o.item()
    if isinstance(o,float) and not np.isfinite(o): return None
    return o

all_results={}
lines=[]
for label,folder in [('observed',HERE),('complete',HERE/'complete_case')]:
    assert (folder/'private_local_only'/'models'/'all_selected_RI.rds').exists(), 'Fit run has not completed'
    comp=pd.read_csv(folder/'model_comparison.csv')
    init=pd.read_csv(folder/'model_initialization_diagnostics.csv')
    qual=pd.read_csv(folder/'classification_quality.csv')
    classes=pd.read_csv(folder/'class_summary.csv')
    assert comp.conv.eq(1).all()
    assert comp.n_subjects.eq(10810 if label=='observed' else 6826).all()
    assert set(init.k)=={1,2,3,4}
    attempts=[]
    for k in range(2,5):
        sh=init[(init.k==k)&(init.phase=='short')]
        fu=init[(init.k==k)&(init.phase=='full')]
        assert len(sh)==40 and len(fu)==6
        attempts.append({'k':k,'short_attempts':len(sh),'short_converged':int(sh.conv.eq(1).sum()),
                         'full_attempts':len(fu),'full_converged':int(fu.conv.eq(1).sum()),
                         'full_final_statuses':fu.conv.astype(int).tolist()})
    best=int(comp.loc[comp.BIC.idxmin(),'k'])
    record={'label':label,'model_comparison':comp.to_dict('records'),'initialization_attempts':attempts,
            'best_BIC_k_among_valid_tested':best,'valid_converged_classes':comp.k.astype(int).tolist(),
            'classification_quality':qual.to_dict('records'),'class_summary':classes.to_dict('records')}
    all_results[label]=record
    lines.append(f'{label.upper()} DATA: BIC comparison among converged tested K=1-4 models; no mortality outcome used.')
    for r in comp.itertuples():
        lines.append(f'K={r.k}: n={r.n_subjects}, observations={r.n_measurements}, loglik={r.loglik:.6f}, AIC={r.AIC:.3f}, BIC={r.BIC:.3f}, conv={r.conv}; best LL repeated in {r.best_loglik_repeats_tolerance_1e_4}/{r.full_converged_starts} fully converged refits (tolerance 1e-4).')
    lines.append(f'Lowest BIC among valid tested models: K={best}. This is not a globally optimal class-count claim.')
    for db in ['MIMIC-IV','eICU']:
        q=qual[(qual.k==best)&(qual.dataset==db)].iloc[0]
        cs=classes[(classes.k==best)&(classes.dataset==db)].sort_values('class')
        lines.append(f'{db}, K={best}: n={int(q.n)}, entropy={q.entropy:.4f}, mean maximum posterior={q.mean_max_posterior:.4f}, maximum posterior <0.8 in {100*q.proportion_max_posterior_lt_0_8:.2f}%; smallest assigned class={int(q.min_assigned_class_n)} ({q.min_assigned_class_pct:.2f}%).')
        lines.append('Assigned class sizes: '+', '.join(f'C{int(x["class"])}={int(x["assigned_n"])} ({x["assigned_pct"]:.2f}%, APP {x["average_posterior_assigned"]:.4f})' for x in cs.to_dict('records'))+'.')
    lines.append('')
methods=('Formal Gaussian latent class linear mixed models were fitted in R 4.6.1 using lcmm 2.2.2 (hlme), with SIC as the longitudinal response and time coded 0, 1 and 2 for W1-W3. Each class had an intercept and linear time slope; a subject random intercept and residual variance were shared across classes. Class probabilities had intercepts only. Candidate K=1-4 models were evaluated without loading mortality outcomes. Each K>1 used 40 seeded random starts from the one-class model (up to 30 iterations); the six highest short-run likelihoods were refitted to at most 1,000 iterations. Parameter, likelihood and derivative convergence tolerances were all 1e-5; only full convergence (conv=1) was accepted for model comparison. BIC used subjects as the sample size. Posterior assignments used maximum probability; neutral labels were sorted by fitted W1 level. The complete-three-window subset repeated the same specification and search. Frozen derivation models were applied to observed eICU SIC values for descriptive posterior allocation, not mortality-prediction validation.\n')
limitations=('SIC is an ordered, bounded 0-6 score, whereas hlme uses a Gaussian continuous approximation. Classes can approximate distributional non-normality rather than distinct biological populations. Three windows support only the specified short linear trajectories. The full observed-data analysis included 1,341 MIMIC admissions with one observed SIC value (12.4%); these inform level but not individual change. Observed-value likelihood requires an unverified missing-at-random assumption. The complete-case analysis selects a different observation population and cannot eliminate informative missingness. eICU posterior allocation is descriptive and includes repeated admissions from some persons. This univariate SIC model differs from the original multivariable K-means feature space and does not reproduce or validate its K4 labels. A BIC minimum at K4 means best among K1-K4 candidates, not proof that four classes are globally optimal, biologically real or treatment-relevant. All patient-level records and fitted model parameters remain local and restricted; this public export contains only source code and aggregate reporting.\n')
all_results['methods']=methods
all_results['limitations']=limitations
all_results['source_urls']=['https://www.jstatsoft.org/article/view/v078i02','https://cecileproust-lima.github.io/lcmm/reference/hlme.html','https://cecileproust-lima.github.io/lcmm/reference/gridsearch.html','https://cecileproust-lima.github.io/lcmm/reference/summaryplot.html']
(HERE/'FINAL_LATENT_REPORT.json').write_text(json.dumps(clean(all_results),indent=2,ensure_ascii=False),encoding='utf-8')
(HERE/'LATENT_METHODS_RESULTS.txt').write_text(methods+'\n'+'\n'.join(lines)+'\n'+limitations,encoding='utf-8')
print('\n'.join(lines))
