"""Post hoc sensitivity analyses using local, temporally eligible frozen cohorts.
Never downloads raw data. Outputs ending RESTRICTED contain participant rows.
Independent of main analysis; all uncertainty conditional on fitted predictions.
"""
from pathlib import Path
import os,sys,json,warnings
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'_deps'))
import revision_analysis as r
import numpy as np,pandas as pd
from sklearn.base import BaseEstimator,TransformerMixin,ClassifierMixin
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer,SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold,GridSearchCV
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,log_loss
from sklearn.utils.validation import check_is_fitted
from scipy.special import expit,logit
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits
import joblib
OUT=HERE/'sensitivity_revision'
OUT.mkdir(exist_ok=True)
M3=r.BASE+[r.SOFA[0],r.SIC[0]]+r.SUMMARY
AUX=r.BASE+r.SIC+r.LABS+r.SOFA
PROP=r.BASE+[r.SOFA[0],r.SIC[0]]+[c for c in r.LABS if c.startswith('w1_')]

def complete(X):return X[r.SIC].notna().all(axis=1).to_numpy()
def write_json(path,x):r.js(path,x)
def summary_values(X):
    s=X[r.SIC].to_numpy(float)
    return np.column_stack([X[r.BASE+[r.SOFA[0],r.SIC[0]]].to_numpy(float),s.mean(axis=1),s.max(axis=1),s.min(axis=1),(s[:,2]-s[:,0])/2])

class IterativeM3(BaseEstimator,TransformerMixin):
    """Predictor-only deterministic imputation, then recompute SIC summaries."""
    def __init__(self,max_iter=20):self.max_iter=max_iter
    def fit(self,X,y=None):
        # Standardizing observed values first makes iterative regression less scale-sensitive.
        self.input_scaler_=StandardScaler().fit(X[AUX])
        z=self.input_scaler_.transform(X[AUX])
        lower=np.full(len(AUX),-np.inf);upper=np.full(len(AUX),np.inf)
        for j,c in enumerate(AUX):
            if c in r.SIC:
                lower[j]=(0-self.input_scaler_.mean_[j])/self.input_scaler_.scale_[j]
                upper[j]=(6-self.input_scaler_.mean_[j])/self.input_scaler_.scale_[j]
        self.imputer_=IterativeImputer(max_iter=self.max_iter,tol=0.01,random_state=42,sample_posterior=False,initial_strategy='median',skip_complete=True,min_value=lower,max_value=upper,keep_empty_features=True)
        zi=self.imputer_.fit_transform(z)
        a=pd.DataFrame(self.input_scaler_.inverse_transform(zi),columns=AUX,index=X.index)
        self.output_scaler_=StandardScaler().fit(summary_values(a))
        return self
    def transform(self,X):
        z=self.imputer_.transform(self.input_scaler_.transform(X[AUX]))
        a=pd.DataFrame(self.input_scaler_.inverse_transform(z),columns=AUX,index=X.index)
        return self.output_scaler_.transform(summary_values(a))

class WeightedCompleteM3(ClassifierMixin,BaseEstimator):
    """Training-fold propensity + selected-case weighted outcome model."""
    def __init__(self,C=1.0,weight_cap=10.0):self.C=C;self.weight_cap=weight_cap
    def fit(self,X,y):
        sel=complete(X);self.prevalence_=float(sel.mean());self.classes_=np.array([0,1])
        self.propensity_=Pipeline([('imputer',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),('scale',StandardScaler()),('model',LogisticRegression(C=1.0,solver='lbfgs',max_iter=2000))]).fit(X[PROP],sel.astype(int))
        w=self.weights(X)[sel]
        self.matrix_=r.MatrixBuilder(tuple(M3),0).fit(r.features(X.loc[sel]))
        self.outcome_=LogisticRegression(C=self.C,solver='lbfgs',max_iter=2000).fit(self.matrix_.transform(r.features(X.loc[sel])),np.asarray(y)[sel],sample_weight=w)
        return self
    def propensity(self,X):return np.clip(self.propensity_.predict_proba(X[PROP])[:,1],1e-3,1-1e-3)
    def weights(self,X):return np.minimum(self.prevalence_/self.propensity(X),self.weight_cap)
    def predict_proba(self,X):
        return self.outcome_.predict_proba(self.matrix_.transform(r.features(X)))
    def score(self,X,y):
        sel=complete(X)
        return -log_loss(np.asarray(y)[sel],self.predict_proba(X)[sel,1],sample_weight=self.weights(X)[sel],labels=[0,1])

if __name__=='__main__':
    sys.modules['sensitivity_revision']=sys.modules[__name__]
    IterativeM3.__module__='sensitivity_revision'
    WeightedCompleteM3.__module__='sensitivity_revision'

def propensity_score(est,X,y):return est.score(X,y)
def candidate(kind,seed):
    if kind=='iterative':p=Pipeline([('matrix',IterativeM3()),('model',LogisticRegression(solver='lbfgs',max_iter=2000))]);grid={'model__C':[.01,.1,1.,10.]};scoring='neg_log_loss'
    elif kind=='complete':p=Pipeline([('matrix',r.MatrixBuilder(tuple(M3),0)),('model',LogisticRegression(solver='lbfgs',max_iter=2000))]);grid={'model__C':[.01,.1,1.,10.]};scoring='neg_log_loss'
    else:p=WeightedCompleteM3();grid={'C':[.01,.1,1.,10.]};scoring=propensity_score
    return GridSearchCV(p,grid,scoring=scoring,cv=StratifiedKFold(3,shuffle=True,random_state=seed),n_jobs=1,refit=True,error_score='raise')

def weighted_metrics(y,p,w=None):
    y=np.asarray(y);p=np.clip(np.asarray(p),1e-7,1-1e-7)
    w=np.ones(len(y)) if w is None else np.asarray(w)
    x=logit(p);A=np.column_stack([np.ones(len(y)),x])
    def fun(v):
        eta=A@v;q=expit(eta)
        return np.sum(w*(np.logaddexp(0,eta)-y*eta)),A.T@(w*(q-y))
    fit=minimize(fun,[0.,1.],jac=True,method='BFGS',options={'gtol':1e-5})
    a=0.
    for _ in range(40):
        q=expit(x+a);delta=np.sum(w*(y-q))/max(np.sum(w*q*(1-q)),1e-12);a+=delta
        if abs(delta)<1e-9:break
    return {'AUROC':float(roc_auc_score(y,p,sample_weight=w)),'AUPRC':float(average_precision_score(y,p,sample_weight=w)),'Brier':float(np.average((y-p)**2,weights=w)),'calibration_intercept':float(fit.x[0]),'calibration_slope':float(fit.x[1]),'calibration_in_the_large':float(a)}

def evaluate(label,cohort,d,models,baseline,weights=None,B=500):
    y=d.y.to_numpy();groups=[np.array([i]) for i in range(len(d))] if cohort=='MIMIC-IV' else list(d.groupby('uniquepid',sort=False).indices.values())
    rng=np.random.default_rng(42);boot={k:[] for k in models};rows=[];diff=[]
    for bi in range(B):
        ix=np.concatenate([groups[g] for g in rng.integers(0,len(groups),len(groups))])
        for k,p in models.items():boot[k].append(weighted_metrics(y[ix],p[ix],None if weights is None else weights[ix]))
    pts={k:weighted_metrics(y,p,weights) for k,p in models.items()}
    for k,pt in pts.items():
        row={'analysis':label,'cohort':cohort,'model':k,'n':len(y),'events':int(y.sum()),'weighted':weights is not None,'effective_sample_size':len(y) if weights is None else float(weights.sum()**2/np.square(weights).sum())}
        b=pd.DataFrame(boot[k])
        for metric,v in pt.items():row[metric]=v;row[metric+'_lo']=float(b[metric].quantile(.025));row[metric+'_hi']=float(b[metric].quantile(.975))
        rows.append(row)
        if k!=baseline:
            for metric in ['AUROC','AUPRC','Brier']:
                v=np.array([q[metric] for q in boot[k]])-np.array([q[metric] for q in boot[baseline]])
                diff.append({'analysis':label,'cohort':cohort,'model_a':k,'model_b':baseline,'metric':metric,'difference':pt[metric]-pts[baseline][metric],'lo':float(np.quantile(v,.025)),'hi':float(np.quantile(v,.975))})
    print('EVALUATED',label,cohort,{k:round(v['AUROC'],4) for k,v in pts.items()},flush=True)
    return rows,diff

def run():
    # load_data writes audit files: isolate them from the main analysis.
    r.OUT=OUT
    rawm,rawe,m,e=r.load_data()
    internal=pd.read_csv(HERE/'internal_predictions_RESTRICTED.csv')
    external=pd.read_csv(HERE/'external_predictions_RESTRICTED.csv')
    rawm=rawm.set_index('stay_id').loc[internal.stay_id].reset_index();m=r.features(rawm)
    rawe=rawe.set_index('patientunitstayid').loc[external.patientunitstayid].reset_index();e=r.features(rawe)
    assert np.array_equal(rawm.hospital_death.to_numpy(),internal.y.to_numpy())
    assert np.array_equal(rawe.hospital_death.to_numpy(),external.y.to_numpy())
    folds=[(np.flatnonzero(internal.fold.to_numpy()!=k),np.flatnonzero(internal.fold.to_numpy()==k)) for k in range(5)]
    y=internal.y.to_numpy();sm=complete(rawm);se=complete(rawe)
    outm=internal[['stay_id','y','fold']].copy();oute=external[['patientunitstayid','uniquepid','y']].copy()
    outm['observed_all3SIC']=sm;oute['observed_all3SIC']=se
    outm['main_M3']=internal.M3_continuous_SIC;oute['main_M3']=external.M3_continuous_SIC
    tuning=[];weight_reports=[];convergence=[]
    for kind in ['complete','iterative','ipw']:
        print('FIT',kind,flush=True)
        pred=np.full(len(m),np.nan);props=np.full(len(m),np.nan);weights=np.full(len(m),np.nan)
        for k,(tr,va) in enumerate(folds):
            fitidx=tr[sm[tr]] if kind=='complete' else tr
            val=va[sm[va]] if kind=='complete' else va
            model=candidate(kind,100+k)
            X=m if kind=='complete' else rawm
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter('always');model.fit(X.iloc[fitidx],y[fitidx])
            pred[val]=model.predict_proba(X.iloc[val])[:,1]
            tuning.append({'kind':kind,'fold':k,'n_train':len(fitidx),'n_train_complete':int(sm[fitidx].sum()),'best_params':model.best_params_})
            convergence.append({'kind':kind,'fold':k,'warning_count':len(ws),'warnings':sorted(set(str(w.message) for w in ws))})
            if kind=='iterative':convergence[-1]['best_imputer_iterations']=int(model.best_estimator_.named_steps['matrix'].imputer_.n_iter_)
            if kind=='ipw':
                mdl=model.best_estimator_;props[va]=mdl.propensity(rawm.iloc[va]);weights[va]=mdl.weights(rawm.iloc[va])
                weight_reports.append({'fold':k,'train_complete_fraction':mdl.prevalence_,'propensity_min_heldout':float(props[va].min()),'propensity_max_heldout':float(props[va].max()),'heldout_complete_weights_p99':float(np.quantile(weights[va][sm[va]],.99)),'heldout_complete_weights_max':float(weights[va][sm[va]].max()),'heldout_weights_capped':int((weights[va][sm[va]]>=10).sum())})
            print(kind,'fold',k,flush=True)
        X=m if kind=='complete' else rawm;fitidx=np.flatnonzero(sm) if kind=='complete' else np.arange(len(m))
        final=candidate(kind,42)
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always');final.fit(X.iloc[fitidx],y[fitidx])
        val=np.flatnonzero(se) if kind=='complete' else np.arange(len(e));ep=np.full(len(e),np.nan)
        ep[val]=final.predict_proba((e if kind=='complete' else rawe).iloc[val])[:,1]
        outm[kind]=pred;oute[kind]=ep
        tuning.append({'kind':kind,'fold':'full_derivation','n_train':len(fitidx),'best_params':final.best_params_})
        convergence.append({'kind':kind,'fold':'full_derivation','warning_count':len(ws),'warnings':sorted(set(str(w.message) for w in ws))})
        if kind=='iterative':convergence[-1]['best_imputer_iterations']=int(final.best_estimator_.named_steps['matrix'].imputer_.n_iter_)
        if kind=='ipw':
            outm['measurement_propensity_oof']=props;outm['measurement_weight_oof']=weights
            # External evaluation weights must represent the external observation process.
            # Fit each propensity only in the other person-group folds; no outcome in propensity.
            from sklearn.model_selection import GroupKFold
            pe=np.zeros(len(e));we=np.zeros(len(e))
            for k,(tr,va) in enumerate(GroupKFold(5).split(rawe,groups=rawe.uniquepid)):
                pr=Pipeline([('imputer',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),('scale',StandardScaler()),('model',LogisticRegression(C=1.,max_iter=2000))]).fit(rawe.iloc[tr][PROP],se[tr].astype(int))
                pe[va]=np.clip(pr.predict_proba(rawe.iloc[va][PROP])[:,1],1e-3,1-1e-3);we[va]=np.minimum(se[tr].mean()/pe[va],10)
            oute['measurement_propensity_oof']=pe;oute['measurement_weight_oof']=we
            weight_reports.append({'fold':'eICU_groupcrossfit','complete_fraction':float(se.mean()),'propensity_min':float(pe.min()),'propensity_max':float(pe.max()),'complete_weights_p99':float(np.quantile(we[se],.99)),'complete_weights_max':float(we[se].max()),'complete_weights_capped':int((we[se]>=10).sum())})
        joblib.dump(final.best_estimator_,OUT/f'{kind}_M3.joblib')
        outm.to_csv(OUT/'sensitivity_internal_RESTRICTED.csv',index=False);oute.to_csv(OUT/'sensitivity_external_RESTRICTED.csv',index=False)
        write_json(OUT/'tuning.json',tuning);write_json(OUT/'imputation_convergence.json',convergence);write_json(OUT/'measurement_weights.json',weight_reports)
    rows=[];diff=[]
    for cohort,d in [('MIMIC-IV',outm),('eICU',oute)]:
        sel=d.observed_all3SIC.to_numpy();dc=d.loc[sel].copy().reset_index(drop=True)
        for label,z,models,w in [
            ('complete_case_same_evaluation_subset',dc,{'full_cohort_carry_M3':dc.main_M3.to_numpy(),'complete_case_M3':dc.complete.to_numpy()},None),
            ('single_iterative_imputation_whole_cohort',d,{'full_cohort_carry_M3':d.main_M3.to_numpy(),'single_iterative_M3':d.iterative.to_numpy()},None),
            ('IPW_complete_case_evaluation',dc,{'full_cohort_carry_M3':dc.main_M3.to_numpy(),'unweighted_complete_case_M3':dc.complete.to_numpy(),'IPW_complete_case_M3':dc.ipw.to_numpy()},dc.measurement_weight_oof.to_numpy()),
            ('IPW_model_unweighted_complete_case_evaluation',dc,{'unweighted_complete_case_M3':dc.complete.to_numpy(),'IPW_complete_case_M3':dc.ipw.to_numpy()},None)]:
            aa,bb=evaluate(label,cohort,z,models,next(iter(models)),w);rows+=aa;diff+=bb
            pd.DataFrame(rows).to_csv(OUT/'sensitivity_performance_with_95CI.csv',index=False);pd.DataFrame(diff).to_csv(OUT/'sensitivity_paired_differences_with_95CI.csv',index=False)
    # A separate outcome-definition check does not alter the primary hospital endpoint.
    outcome_rows=[]
    alt=(rawe.hospital_death.to_numpy().astype(bool)|rawe.icu_death.to_numpy().astype(bool)).astype(int)
    for model in ['M3_continuous_SIC','Expanded_LR','Expanded_XGB']:
        outcome_rows.append({'analysis':'hospital_OR_unit_death_secondary_external','model':model,'n':len(alt),'events':int(alt.sum()),'discordant_unit_death_hospital_alive':int(((rawe.icu_death==1)&(rawe.hospital_death==0)).sum()),**weighted_metrics(alt,external[model].to_numpy())})
    pd.DataFrame(outcome_rows).to_csv(OUT/'external_composite_outcome_sensitivity.csv',index=False)
    method={'status':'Post hoc reviewer-requested sensitivity analyses','main_cohort_n':{'MIMIC-IV':len(m),'eICU':len(e)},'observed_all3SIC_n':{'MIMIC-IV':int(sm.sum()),'eICU':int(se.sum())},'complete_case':'SIC must be observed in W1,W2,W3 before any imputation. The complete-case-trained M3 is compared with full-cohort-trained carry-forward M3 only on the same complete evaluation cases. With complete SIC, carry-forward leaves the M3 predictors unchanged; this comparison concerns selection/training sensitivity, not an isolated imputation effect.','iterative':'Single deterministic Bayesian-ridge IterativeImputer on standardised predictor-only raw SIC/laboratory/SOFA windows plus age/sex; max_iter20,tol0.01,random_state42,sample_posteriorFalse. Missing SIC constrained to0–6; fractional imputed SIC retained. SIC summaries recomputed after imputation. Standardisation/imputation/classifier tuning fit independently within every inner/outer training fold. No outcome in imputation; no multiple-imputation or pooling claim.','iterative_predictors':AUX,'propensity_covariates':PROP,'IPW':'R=all three raw SIC windows observed. Logistic(C1) propensity model uses baseline-only covariates, median imputation with missingness indicators and scaling inside each fitting fold. Outcome M3 fit only R1 with stabilised weights Pr(R1 in training)/Pr(R1|baseline), cap10; propensity probabilities clipped0.001–0.999. Both propensity and outcome model refit in each inner/outer fold. Inner outcome C chosen by weighted log loss among complete heldout cases. Evaluation uses heldout-fold propensity weights; external observation-propensity models fitted using5 person-group folds in eICU, no external mortality in propensity. This exploratory target-population reweighting depends on baseline-conditional exchangeability and positivity, does not correct missing-not-at-random processes unconditionally, and is not untouched external validation of the observation model.','bootstrap':'500 paired percentile replicates using fixed predictions/weights; MIMIC patient rows and eICU uniquepid clusters. CIs conditional on models, imputations and estimated weights, without pipeline-refitting or hospital-cluster uncertainty.','outcome_sensitivity':'External hospital OR unit death uses frozen original predictions; primary remains original hospital discharge death status. Only point estimates for this explicitly secondary check.','complete_case_weights_effective_n':{'MIMIC-IV':float(outm.loc[sm,'measurement_weight_oof'].sum()**2/np.square(outm.loc[sm,'measurement_weight_oof']).sum()),'eICU':float(oute.loc[se,'measurement_weight_oof'].sum()**2/np.square(oute.loc[se,'measurement_weight_oof']).sum())}}
    write_json(OUT/'sensitivity_methods.json',method)
    (OUT/'sensitivity_methods.txt').write_text('\n\n'.join(f'{k}: {v}' for k,v in method.items()),encoding='utf-8')
    print('DONE sensitivity_revision',flush=True)

def refresh_external():
    """Keep all derivation fits; align and rescore external final eligibility."""
    r.OUT=OUT
    rawm,rawe,m,e=r.load_data()
    external=pd.read_csv(HERE/'external_predictions_RESTRICTED.csv')
    if len(external)!=len(rawe):raise RuntimeError('Wait for main external prediction refresh before running sensitivity refresh')
    rawe=rawe.set_index('patientunitstayid').loc[external.patientunitstayid].reset_index()
    d=pd.read_csv(OUT/'sensitivity_external_RESTRICTED.csv').set_index('patientunitstayid').loc[external.patientunitstayid].reset_index()
    assert np.array_equal(d.y.to_numpy(),external.y.to_numpy())
    assert np.allclose(d.main_M3.to_numpy(),external.M3_continuous_SIC.to_numpy())
    se=complete(rawe)
    from sklearn.model_selection import GroupKFold
    pe=np.zeros(len(d));we=np.zeros(len(d))
    for k,(tr,va) in enumerate(GroupKFold(5).split(rawe,groups=rawe.uniquepid)):
        pr=Pipeline([('imputer',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),('scale',StandardScaler()),('model',LogisticRegression(C=1.,max_iter=2000))]).fit(rawe.iloc[tr][PROP],se[tr].astype(int))
        pe[va]=np.clip(pr.predict_proba(rawe.iloc[va][PROP])[:,1],1e-3,1-1e-3);we[va]=np.minimum(se[tr].mean()/pe[va],10)
    d['measurement_propensity_oof']=pe;d['measurement_weight_oof']=we
    d.to_csv(OUT/'sensitivity_external_RESTRICTED.csv',index=False)
    rows=pd.read_csv(OUT/'sensitivity_performance_with_95CI.csv');rows=rows[rows.cohort.eq('MIMIC-IV')].to_dict('records')
    diff=pd.read_csv(OUT/'sensitivity_paired_differences_with_95CI.csv');diff=diff[diff.cohort.eq('MIMIC-IV')].to_dict('records')
    dc=d.loc[se].copy().reset_index(drop=True)
    for label,z,models,w in [
        ('complete_case_same_evaluation_subset',dc,{'full_cohort_carry_M3':dc.main_M3.to_numpy(),'complete_case_M3':dc.complete.to_numpy()},None),
        ('single_iterative_imputation_whole_cohort',d,{'full_cohort_carry_M3':d.main_M3.to_numpy(),'single_iterative_M3':d.iterative.to_numpy()},None),
        ('IPW_complete_case_evaluation',dc,{'full_cohort_carry_M3':dc.main_M3.to_numpy(),'unweighted_complete_case_M3':dc.complete.to_numpy(),'IPW_complete_case_M3':dc.ipw.to_numpy()},we[se]),
        ('IPW_model_unweighted_complete_case_evaluation',dc,{'unweighted_complete_case_M3':dc.complete.to_numpy(),'IPW_complete_case_M3':dc.ipw.to_numpy()},None)]:
        aa,bb=evaluate(label,'eICU',z,models,next(iter(models)),w);rows+=aa;diff+=bb
    pd.DataFrame(rows).to_csv(OUT/'sensitivity_performance_with_95CI.csv',index=False)
    pd.DataFrame(diff).to_csv(OUT/'sensitivity_paired_differences_with_95CI.csv',index=False)
    outcome_rows=[];alt=(rawe.hospital_death.to_numpy().astype(bool)|rawe.icu_death.to_numpy().astype(bool)).astype(int)
    for model in ['M3_continuous_SIC','Expanded_LR','Expanded_XGB']:
        outcome_rows.append({'analysis':'hospital_OR_unit_death_secondary_external','model':model,'n':len(alt),'events':int(alt.sum()),'discordant_unit_death_hospital_alive':int(((rawe.icu_death==1)&(rawe.hospital_death==0)).sum()),**weighted_metrics(alt,external[model].to_numpy())})
    pd.DataFrame(outcome_rows).to_csv(OUT/'external_composite_outcome_sensitivity.csv',index=False)
    meth=json.loads((OUT/'sensitivity_methods.json').read_text(encoding='utf-8'))
    meth['main_cohort_n']['eICU']=len(d);meth['observed_all3SIC_n']['eICU']=int(se.sum())
    meth['complete_case_weights_effective_n']['eICU']=float(we[se].sum()**2/np.square(we[se]).sum())
    meth['external_refresh']='Excluded records without recorded Alive/Expired hospital discharge status; derivation models unchanged, external observation-propensity fits and conditional bootstrap intervals refreshed on final eligible cohort.'
    meth['iterative_interpretation']='Auxiliary SOFA/laboratory variables contribute to imputed SIC values. Improved discrimination cannot be attributed solely to a better missingness assumption or coagulation-specific information.'
    meth['IPW_interpretation']='Reweights later-window complete measurement within the W1-available landmark cohort. It cannot recover patients excluded because W1 SIC was unavailable or correct nonignorable measurement processes without assumptions.'
    write_json(OUT/'sensitivity_methods.json',meth)
    (OUT/'sensitivity_methods.txt').write_text('\n\n'.join(f'{k}: {v}' for k,v in meth.items()),encoding='utf-8')
    wr=json.loads((OUT/'measurement_weights.json').read_text(encoding='utf-8'))
    wr=[z for z in wr if z.get('fold')!='eICU_groupcrossfit']
    wr.append({'fold':'eICU_groupcrossfit','n':len(d),'complete_fraction':float(se.mean()),'propensity_min':float(pe.min()),'propensity_max':float(pe.max()),'complete_weights_p99':float(np.quantile(we[se],.99)),'complete_weights_max':float(we[se].max()),'complete_weights_capped':int((we[se]>=10).sum())})
    write_json(OUT/'measurement_weights.json',wr)
    print('DONE external sensitivity refresh',len(d),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):
        if '--refresh-external' in sys.argv:refresh_external()
        else:run()
