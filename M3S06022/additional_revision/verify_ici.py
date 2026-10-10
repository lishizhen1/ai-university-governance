"""Independent arithmetic/refit checks for the added ICI analysis."""
from pathlib import Path
import os,sys,json,hashlib
os.environ.setdefault('OMP_NUM_THREADS','2')
HERE=Path(__file__).resolve().parent;BASE=HERE.parent/'audit_analysis'
sys.path.insert(0,str(BASE/'_deps'))
import numpy as np,pandas as pd
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def independent_metrics(y,p):
    z=np.clip(p,1e-6,1-1e-6)
    x=np.log(z/(1-z)).reshape(-1,1)
    s=SplineTransformer(n_knots=4,degree=3,knots='quantile',include_bias=False)
    xx=s.fit_transform(x)
    fit=LogisticRegression(C=1,max_iter=2000,solver='lbfgs',tol=1e-4).fit(xx,y)
    delta=np.abs(fit.predict_proba(xx)[:,1]-p)
    return np.array([sum(delta)/len(delta),np.median(delta),np.percentile(delta,90,method='linear')])

def run():
    table=pd.read_csv(HERE/'ici_e50_e90_with_95CI.csv')
    boot=pd.read_csv(HERE/'ici_bootstrap_metric_replicates.csv')
    differences=pd.read_csv(HERE/'ici_e50_e90_paired_differences.csv')
    keys=['ICI','E50','E90'];results=[];cluster_check=True
    assert len(table)==26 and len(boot)==13000 and len(differences)==72
    for cohort,file in [('MIMIC-IV','internal_predictions_RESTRICTED.csv'),('eICU','external_predictions_RESTRICTED.csv')]:
        d=pd.read_csv(BASE/file);y=d.y.to_numpy()
        rng=np.random.default_rng(20261010)
        if cohort=='MIMIC-IV':idx=rng.integers(0,len(d),len(d))
        else:
            gs=list(d.groupby('uniquepid',sort=False).indices.values())
            idx=np.concatenate([gs[k] for k in rng.integers(0,len(gs),len(gs))])
            multiplicity=np.bincount(idx,minlength=len(d))
            assert all(np.ptp(multiplicity[g])==0 for g in gs)
        for _,row in table[table.cohort.eq(cohort)].iterrows():
            p=d[row.model].to_numpy();point=independent_metrics(y,p)
            err=float(np.max(np.abs(point-row[keys].to_numpy(float))))
            assert err<1e-12
            sampled=independent_metrics(y[idx],p[idx])
            target=boot[(boot.cohort==cohort)&(boot.model==row.model)&(boot.bootstrap==0)].iloc[0]
            booterr=float(np.max(np.abs(sampled-target[keys].to_numpy(float))))
            assert booterr<1e-12 and target.n_resampled_stays==len(idx)
            bb=boot[(boot.cohort==cohort)&(boot.model==row.model)]
            assert len(bb)==500
            for key in keys:
                lo,hi=np.quantile(bb[key],[.025,.975],method='linear')
                assert abs(lo-row[key+'_lo'])<1e-12 and abs(hi-row[key+'_hi'])<1e-12
            assert row.E50<=row.E90
            results.append({'cohort':cohort,'model':row.model,'independent_point_max_error':err,'independent_bootstrap0_max_error':booterr,'confidence_endpoints_reconstructed':True})
    for _,row in differences.iterrows():
        a=boot[(boot.cohort==row.cohort)&(boot.model==row.model_a)].sort_values('bootstrap')[row.metric].to_numpy()
        b=boot[(boot.cohort==row.cohort)&(boot.model==row.model_b)].sort_values('bootstrap')[row.metric].to_numpy()
        lo,hi=np.quantile(a-b,[.025,.975],method='linear')
        assert abs(lo-row.lo)<1e-12 and abs(hi-row.hi)<1e-12
    snapshot=json.loads((BASE/'FINAL_SNAPSHOT.json').read_text(encoding='utf-8'))
    checked=[]
    for rel,h in snapshot['sha256'].items():
        if Path(rel).suffix in ['.csv','.joblib']:
            p=BASE/rel
            assert p.exists() and sha(p)==h,rel
            checked.append(rel)
    forbidden={'subject_id','hadm_id','stay_id','patientunitstayid','uniquepid','hospitalid','y','probability','prediction'}
    for name in ['ici_e50_e90_with_95CI.csv','ici_e50_e90_paired_differences.csv','ici_bootstrap_metric_replicates.csv','ici_point_estimates.csv']:
        assert not forbidden.intersection(pd.read_csv(HERE/name,nrows=0).columns)
    out={'status':'PASS','independent_refits':results,'paired_interval_count_reconstructed':len(differences),'person_cluster_membership_multiplicity_check':True,'old_numeric_and_model_files_matching_frozen_hashes':len(checked),'old_numeric_and_model_files':checked,'no_participant_identifier_or_prediction_columns_exported':True,'original_analysis_snapshot_sha256':sha(BASE/'FINAL_SNAPSHOT.json'),'source_script_sha256':sha(HERE/'ici_analysis.py')}
    (HERE/'ici_independent_verification.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print('PASS independent refits',len(results),'paired CIs',len(differences),'unchanged original numeric/model files',len(checked))
if __name__=='__main__':
    with threadpool_limits(limits=2):run()
