"""Additional fixed-prediction calibration summaries; original artifacts read-only.

Primary estimator exactly follows the Figure 2 logistic B-spline specification.
No participant-level data, fitted calibrator objects or fitted parameters are saved.
Run: python ici_analysis.py [--bootstrap 500]
Requires the parent analysis environment and authorized local prediction inputs.
"""
from pathlib import Path
import os,sys,json,hashlib,argparse,warnings,time
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
HERE=Path(__file__).resolve().parent
ANALYSIS=HERE.parent/'audit_analysis'
sys.path.insert(0,str(ANALYSIS/'_deps'))
import numpy as np
import pandas as pd
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

SEED=20261010
EPSILON=1e-6
METRICS=['ICI','E50','E90']
PAIRS=[
 ('M2_admission_SIC','M1_admission_SOFA'),
 ('M3_continuous_SIC','M2_admission_SIC'),
 ('M3_continuous_SIC','Peak_SOFA'),
 ('Peak_SOFA_plus_SIC','Peak_SOFA'),
 ('Coagulation_only_serial','M3_continuous_SIC'),
 ('Serial_SOFA','M3_continuous_SIC'),
 ('M3_plus_K2','M3_continuous_SIC'),
 ('M3_plus_K4_exploratory','M3_continuous_SIC'),
 ('Expanded_LR','M3_continuous_SIC'),
 ('Expanded_XGB','Expanded_LR'),
 ('M3_missing_indicators','M3_continuous_SIC'),
 ('M3_continuous_SIC','M3_level_only'),
]
REFERENCE={
 'authors':['Peter C. Austin','Ewout W. Steyerberg'],
 'title':'The Integrated Calibration Index (ICI) and related metrics for quantifying the calibration of logistic regression models',
 'journal':'Statistics in Medicine','year':2019,'volume':38,'issue':21,'pages':'4051–4065',
 'doi':'10.1002/sim.8281','pmid':'31270850',
 'primary_url':'https://onlinelibrary.wiley.com/doi/full/10.1002/sim.8281',
 'semantic_scholar_paper_id':'dd4bb722ff1661942c64d0c6e62fa3d9fe526a40',
}

def js(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def validate(y,p):
    y=np.asarray(y,dtype=float);p=np.asarray(p,dtype=float)
    if y.ndim!=1 or p.ndim!=1 or len(y)!=len(p) or not len(y):raise ValueError('Nonempty aligned 1D vectors required')
    if not np.isfinite(y).all() or not np.isfinite(p).all():raise ValueError('Missing or nonfinite values are not silently removed')
    if not np.isin(y,[0,1]).all():raise ValueError('Outcomes must be binary')
    if np.any((p<0)|(p>1)):raise ValueError('Predictions must be in [0,1]')
    return y,p

def error_summaries(p,calibrated):
    p=np.asarray(p,float);calibrated=np.asarray(calibrated,float)
    if p.shape!=calibrated.shape or not np.isfinite(calibrated).all():raise ValueError('Invalid calibrated values')
    if np.any((calibrated<0)|(calibrated>1)):raise ValueError('Calibrated probabilities outside [0,1]')
    delta=np.abs(calibrated-p)
    return {'ICI':float(np.mean(delta)),
            'E50':float(np.quantile(delta,.5,method='linear')),
            'E90':float(np.quantile(delta,.9,method='linear'))}

class SplineCalibration:
    """Figure 2 specification; explicit degeneracy policy, no tuning."""
    def fit(self,y,p):
        y,p=validate(y,p)
        self.mode_='quantile_spline'
        self.convergence_warning_=False
        self.n_iter_=0
        z=np.clip(p,EPSILON,1-EPSILON)
        x=np.log(z/(1-z)).reshape(-1,1)
        if len(np.unique(y))==1 or np.ptp(x)==0:
            self.mode_='constant_degenerate'
            self.event_rate_=float(y.mean())
            return self
        knots=np.quantile(x[:,0],np.linspace(0,1,4),method='linear')
        if np.any(np.diff(knots)<=0):
            # Predetermined numerical boundary handling only. Never outcome-selected.
            self.mode_='uniform_knots_due_to_ties'
            knot_spec=np.linspace(float(x.min()),float(x.max()),4).reshape(-1,1)
        else:knot_spec='quantile'
        self.spline_=SplineTransformer(n_knots=4,degree=3,knots=knot_spec,include_bias=False)
        xx=self.spline_.fit_transform(x)
        self.lr_=LogisticRegression(C=1,max_iter=2000,solver='lbfgs',tol=1e-4)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',ConvergenceWarning)
            self.lr_.fit(xx,y)
        self.convergence_warning_=any(issubclass(w.category,ConvergenceWarning) for w in caught)
        self.n_iter_=int(self.lr_.n_iter_.max())
        if self.convergence_warning_:raise RuntimeError('Calibrator failed to converge; no replicate silently dropped')
        return self
    def predict(self,p):
        p=np.asarray(p,float)
        if self.mode_=='constant_degenerate':return np.full(len(p),self.event_rate_)
        z=np.clip(p,EPSILON,1-EPSILON)
        x=np.log(z/(1-z)).reshape(-1,1)
        return self.lr_.predict_proba(self.spline_.transform(x))[:,1]

def evaluate(y,p):
    calibrator=SplineCalibration().fit(y,p)
    return error_summaries(p,calibrator.predict(p)),calibrator

def boundary_checks():
    tests=[]
    p=np.array([.1,.2,.3,.4]);c=np.array([.1,.3,.1,.8]);d=np.array([0,.1,.2,.4])
    m=error_summaries(p,c)
    assert np.isclose(m['ICI'],.175) and np.isclose(m['E50'],.15) and np.isclose(m['E90'],.34)
    tests.append({'test':'hand-computed empirical mean/median/90th-percentile','passed':True})
    perfect=error_summaries(p,p)
    assert all(v==0 for v in perfect.values())
    tests.append({'test':'identity calibration gives zero for all summaries','passed':True})
    y=np.array([0,1]*5);p=np.full(10,.5);m,cal=evaluate(y,p)
    assert all(v==0 for v in m.values()) and cal.mode_=='constant_degenerate'
    tests.append({'test':'constant prediction with matching prevalence','passed':True})
    y=np.ones(10);p=np.zeros(10);m,cal=evaluate(y,p)
    assert all(v==1 for v in m.values())
    tests.append({'test':'single-class opposite-boundary predictions yield error one','passed':True})
    p=np.array([0.,0.,1.,1.,.2,.4,.6,.8]);y=np.array([0,1,1,0,0,1,0,1]);m,cal=evaluate(y,p)
    assert all(np.isfinite(v) and 0<=v<=1 for v in m.values())
    tests.append({'test':'exact zero/one predictions remain finite; clipping only transforms logit','passed':True})
    p=np.array([.2]*12+[.8]*8);y=np.array([0,1]*10);m,cal=evaluate(y,p)
    assert cal.mode_=='uniform_knots_due_to_ties' and all(0<=v<=1 for v in m.values())
    tests.append({'test':'tied quantile knots use predetermined numerical fallback','passed':True})
    for yy,pp in [(np.array([0,1]),np.array([.2,np.nan])),(np.array([0,1]),np.array([.2,1.1])),(np.array([0,2]),np.array([.2,.8]))]:
        try:validate(yy,pp)
        except ValueError:pass
        else:raise AssertionError('Invalid input was accepted')
    tests.append({'test':'invalid outcomes/probabilities/NaN raise instead of deleting records','passed':True})
    # Check resampling whole clusters preserves all associated stays each draw.
    groups=[np.array([0,1]),np.array([2]),np.array([3,4,5])]
    draws=np.array([2,0,2]);idx=np.concatenate([groups[g] for g in draws])
    assert np.array_equal(idx,[3,4,5,0,1,3,4,5])
    tests.append({'test':'person-cluster resampling carries repeated stays together','passed':True})
    return tests

def plan(B):
    return {
      'analysis':'Additional ICI/E50/E90 for frozen primary predictions',
      'definition':'ICI=mean(abs(c_hat(p_i)-p_i)); E50 and E90 are empirical quantiles of the same absolute deviations; lower is better.',
      'reference':REFERENCE,
      'estimator_fixed_before_reading_results':{'predictor':'logit(p), clipped to [1e-6,1−1e-6] only for transform','SplineTransformer':{'n_knots':4,'degree':3,'knots':'quantile','include_bias':False},'LogisticRegression':{'C':1,'max_iter':2000,'solver':'lbfgs','tol':1e-4},'selection':'Exact pre-existing Figure 2 specification; no result-dependent selection.'},
      'evaluation':'All observed prediction values; equal weight per ICU admission; no trimming to the 1st–99th plotting interval, no uniform-grid averaging, and no modification of frozen predictions.',
      'quantile_method':'numpy linear interpolation for E50/E90 and percentile confidence intervals',
      'bootstrap':{'replicates':B,'seed':SEED,'MIMIC-IV':'sample n independent first-stay people with replacement','eICU':'sample unique people with replacement and carry all included stays each time','calibrator':'Refit quantile knots, spline and logistic coefficients in every bootstrap sample; assess at the resampled observed predictions.','pairing':'Identical bootstrap draw for all 13 models within cohort','CI':'2.5th and97.5th percentile; no bootstrap replicate silently discarded','conditional_limitation':'Original risk model fitting/tuning and OOF generation are not repeated. Smoother-estimation uncertainty is included; model-development uncertainty and eICU between-hospital heterogeneity are not fully included.'},
      'distinction_from_original_paper':'Original paper illustrated loess and restricted cubic splines, and used a derivation+validation bootstrap refitting the risk model. Here a fixed logistic B-spline estimator is used for consistency with our existing plot, and intervals explicitly condition on the risk predictions.',
      'numerical_boundary_policy':'Single outcome class or constant prediction: constant empirical event-rate curve. Duplicated quantile knots with varying predictions: four equally spaced logit knots. Neither policy depends on observed ICI; frequency is reported. Original risk p is retained in absolute-error calculation.',
      'model_pairs':PAIRS,
      'claims':'Post hoc descriptive addition; intervals unadjusted for multiplicity; no clinical-calibration threshold or deployment claim.',
    }

def run(B=500):
    HERE.mkdir(parents=True,exist_ok=True)
    js(HERE/'ici_analysis_plan.json',plan(B))
    checks=boundary_checks()
    paths={cohort:ANALYSIS/name for cohort,name in [('MIMIC-IV','internal_predictions_RESTRICTED.csv'),('eICU','external_predictions_RESTRICTED.csv')]}
    input_hash={path.name:sha(path) for path in paths.values()}
    snapshot=json.loads((ANALYSIS/'FINAL_SNAPSHOT.json').read_text(encoding='utf-8'))
    for path in paths.values():
        if path.name in snapshot['sha256']:assert input_hash[path.name]==snapshot['sha256'][path.name], 'Frozen input changed'
    models=list(json.loads((ANALYSIS/'feature_sets.json').read_text(encoding='utf-8')))
    assert len(models)==13
    expected={'MIMIC-IV':(10810,2174),'eICU':(4355,979)}
    ref_curves=ANALYSIS.parent/'figures'/'figure2_calibration_spline_source.csv'
    saved_curves=pd.read_csv(ref_curves) if ref_curves.exists() else None
    rows=[];point_rows=[];difference_rows=[];boot_rows=[];qa=[];consistency=[]
    start=time.time()
    for cohort,path in paths.items():
        df=pd.read_csv(path);y=df.y.to_numpy(int);pred=df[models].to_numpy(float)
        n=len(df);assert (n,int(y.sum()))==expected[cohort]
        if cohort=='MIMIC-IV':
            assert df.stay_id.is_unique
            groups=None;ng=n
        else:
            assert df.uniquepid.notna().all()
            groups=list(df.groupby('uniquepid',sort=False).indices.values());ng=len(groups)
        point={}
        modes={m:{} for m in models};nit={m:[] for m in models}
        for j,name in enumerate(models):
            val,cal=evaluate(y,pred[:,j]);point[name]=val
            point_rows.append({'cohort':cohort,'model':name,'n':n,'events':int(y.sum()),'unique_people':ng,**val})
            if saved_curves is not None:
                ref=saved_curves[(saved_curves.cohort==cohort)&(saved_curves.model==name)]
                if len(ref):
                    pp=ref.predicted.to_numpy(float);yy=ref.smoothed_observed.to_numpy(float)
                    error=float(np.max(np.abs(cal.predict(pp)-yy)))
                    assert error<1e-6,(cohort,name,error)
                    consistency.append({'cohort':cohort,'model':name,'stored_curve_points':len(ref),'max_absolute_difference':error,'passed':True})
            modes[name][cal.mode_]=1;nit[name].append(cal.n_iter_)
        pd.DataFrame(point_rows).to_csv(HERE/'ici_point_estimates.csv',index=False)
        samples=np.zeros((B,len(models),len(METRICS)))
        rng=np.random.default_rng(SEED)
        for b in range(B):
            draw=rng.integers(0,ng,ng)
            idx=draw if groups is None else np.concatenate([groups[g] for g in draw])
            yy=y[idx]
            for j,name in enumerate(models):
                val,cal=evaluate(yy,pred[idx,j])
                samples[b,j]=[val[k] for k in METRICS]
                modes[name][cal.mode_]=modes[name].get(cal.mode_,0)+1;nit[name].append(cal.n_iter_)
                boot_rows.append({'cohort':cohort,'model':name,'bootstrap':b,'n_resampled_stays':len(idx),**val})
            if b%50==0:print(cohort,'bootstrap',b,'/',B,'elapsed_s',round(time.time()-start),flush=True)
        for j,name in enumerate(models):
            rr={'cohort':cohort,'model':name,'n':n,'events':int(y.sum()),'unique_people':ng,'bootstrap_replicates':B}
            for k,metric in enumerate(METRICS):
                rr[metric]=point[name][metric]
                rr[metric+'_lo']=float(np.quantile(samples[:,j,k],.025,method='linear'))
                rr[metric+'_hi']=float(np.quantile(samples[:,j,k],.975,method='linear'))
            rows.append(rr)
            qa.append({'cohort':cohort,'model':name,'calibrator_fit_count_including_point':B+1,'fit_modes':modes[name],'maximum_solver_iterations':max(nit[name]),'failed_fits':0,'epsilon_clipped_original_predictions':int(((pred[:,j]<EPSILON)|(pred[:,j]>1-EPSILON)).sum())})
        for a,bname in PAIRS:
            ia=models.index(a);ib=models.index(bname)
            for j,metric in enumerate(METRICS):
                d=samples[:,ia,j]-samples[:,ib,j]
                difference_rows.append({'cohort':cohort,'model_a':a,'model_b':bname,'metric':metric,'difference':point[a][metric]-point[bname][metric],'lo':float(np.quantile(d,.025,method='linear')),'hi':float(np.quantile(d,.975,method='linear')),'bootstrap_replicates':B})
        pd.DataFrame(rows).to_csv(HERE/'ici_e50_e90_with_95CI.csv',index=False)
        pd.DataFrame(difference_rows).to_csv(HERE/'ici_e50_e90_paired_differences.csv',index=False)
        pd.DataFrame(boot_rows).to_csv(HERE/'ici_bootstrap_metric_replicates.csv',index=False)
    final_hash={path.name:sha(path) for path in paths.values()}
    assert input_hash==final_hash
    assert len(rows)==26 and len(boot_rows)==2*len(models)*B
    assert np.isfinite(pd.DataFrame(rows)[[m+s for m in METRICS for s in ['','_lo','_hi']]].to_numpy()).all()
    js(HERE/'ici_verification.json',{'boundary_and_definition_tests':checks,'agreement_with_existing_Figure2':consistency,'calibrator_diagnostics':qa,'input_sha256_before':input_hash,'input_sha256_after':final_hash,'original_inputs_unchanged':True,'total_calibrator_fits':len(qa)*(B+1),'total_bootstrap_rows':len(boot_rows),'saved_individual_predictions':False,'saved_calibrator_parameters':False,'elapsed_seconds':time.time()-start})
    import sklearn,scipy,platform
    js(HERE/'ici_environment.json',{'Python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scikit_learn':sklearn.__version__,'scipy':scipy.__version__,'thread_limit':2,'bootstrap_seed':SEED})
    print('ICI COMPLETE',len(rows),'model-cohort estimates',len(difference_rows),'paired metric contrasts',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--bootstrap',type=int,default=500);args=parser.parse_args()
    if args.bootstrap<100:raise ValueError('Use at least100 bootstrap replicates')
    with threadpool_limits(limits=2):run(args.bootstrap)
