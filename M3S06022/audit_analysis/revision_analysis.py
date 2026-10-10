"""Reviewer-requested reanalysis; no original data/output is modified.
Run with bundled Python; local package dependencies live in ./_deps.
Revised analyses harmonize SIC INR boundary (>1.2) and positivity (total>=4 and coag>=3).
Bootstrap intervals condition on cross-fitted predictions; they do not re-fit the pipeline.
"""
from pathlib import Path
import os, sys, json, hashlib, platform, warnings
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
HERE=Path(__file__).resolve().parent
if __name__=='__main__': sys.modules['revision_analysis']=sys.modules[__name__]
sys.path.insert(0,str(HERE/'_deps'))
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, silhouette_score, adjusted_rand_score
from threadpoolctl import threadpool_limits
import joblib
ROOT=Path(os.environ.get('SIC_PROJECT_ROOT',str(HERE.parents[1])))
DATA=Path(os.environ.get('SIC_DATA_ROOT',str(ROOT/'03_数据源')))
MC=DATA/'MIMIC-IV'
EC=DATA/'eICU'/'08_full_sofa_sepsis3_eicu'
OUT=HERE
IDS=['subject_id','hadm_id','stay_id']
SIC=[f'w{i}_sic_total_score' for i in range(1,4)]
SOFA=[f'w{i}_sofa_total_assuming_missing_normal' for i in range(1,4)]
COAG=[f'w{i}_{v}' for v in ['platelet_min','inr_max'] for i in range(1,4)]
COMP=[f'w{i}_sofa_{v}' for v in ['respiration','coagulation','liver','cardiovascular','cns','renal'] for i in range(1,4)]
CLUSTER=SIC+COAG+[f'w{i}_{v}' for v in ['sofa_4systems_for_sic','sofa_respiration','sofa_cardiovascular','sofa_renal'] for i in range(1,4)]
SUMMARY=['sic_mean_3w','sic_max_3w','sic_min_3w','sic_slope']
BASE=['anchor_age','gender_numeric']
LABS=[f'w{i}_{v}' for v in ['platelet_min','inr_max','bilirubin_total_max','creatinine_max','lactate_max'] for i in range(1,4)]

def b(s): return s.astype(str).str.lower().isin(['true','t','1','yes'])
def js(path,obj): path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=lambda x: x.item() if hasattr(x,'item') else str(x)),encoding='utf-8')
def pivot(d,key,cols):
    p=d.pivot(index=key,columns='window_name',values=cols)
    p.columns=[f'{w[:2]}_{v}' for v,w in p.columns]
    return p.reset_index()
def harmonize(d):
    d=d.copy()
    for i in range(1,4):
        p=d[f'w{i}_platelet_min']; r=d[f'w{i}_inr_max']; s=d[f'w{i}_sofa_4systems_for_sic']
        pc=pd.Series(np.select([p<100,p<150],[2.,1.],0.),index=d.index).where(p.notna())
        rc=pd.Series(np.select([r>1.4,r>1.2],[2.,1.],0.),index=d.index).where(r.notna())
        oc=pd.Series(np.select([s>=2,s==1],[2.,1.],0.),index=d.index).where(s.notna())
        d[f'w{i}_sic_platelet_component']=pc; d[f'w{i}_sic_inr_component']=rc; d[f'w{i}_sic_organ_sofa_component']=oc
        d[f'w{i}_sic_total_score']=pc+rc+oc
        d[f'w{i}_sic_flag']=((pc+rc+oc>=4)&(pc+rc>=3)).where(p.notna()&r.notna())
    return d
def features(d):
    d=d.copy()
    # Patient-local W1 carry-forward; uses neither other patients nor outcomes.
    carry=SIC+LABS+COMP+[f'w{i}_sofa_4systems_for_sic' for i in range(1,4)]
    for c in carry:
        d[c+'_missing']=d[c].isna().astype(int)
        if c.startswith(('w2_','w3_')): d[c]=d[c].fillna(d['w1_'+c[3:]])
    d['sic_mean_3w']=d[SIC].mean(axis=1);d['sic_max_3w']=d[SIC].max(axis=1);d['sic_min_3w']=d[SIC].min(axis=1)
    d['sic_slope']=(d[SIC[2]]-d[SIC[0]])/2
    d['sofa_peak']=d[SOFA].max(axis=1)
    return d
def desc(d,name,age='anchor_age',sex='gender_numeric',los='icu_los_days'):
    z={'group':name,'n':len(d),'hospital_deaths':int(d['hospital_death'].sum()),'hospital_mortality_pct':float(d['hospital_death'].mean()*100)}
    for col,label in [(age,'age'),(los,'ICU_LOS'),('w1_sofa_total_assuming_missing_normal','W1_SOFA'),('w1_sic_total_score','W1_SIC')]:
        z[label+'_median']=float(d[col].median()); z[label+'_q1']=float(d[col].quantile(.25));z[label+'_q3']=float(d[col].quantile(.75))
    z['female_pct']=float((1-d[sex]).mean()*100)
    return z
def eicu_text_by_landmark(ev):
    eligible=set(ev.patientunitstayid.astype(int));diag_ids=set();boundary_ids=set();ambiguous_ids=set();terms={}
    path=DATA/'eicu-collaborative-research-database-2.0'/'diagnosis.csv.gz'
    for ch in pd.read_csv(path,usecols=['patientunitstayid','diagnosisstring','icd9code','diagnosisoffset'],chunksize=500000,low_memory=False):
        offset=pd.to_numeric(ch.diagnosisoffset,errors='coerce')
        ch=ch.loc[ch.patientunitstayid.isin(eligible)&offset.notna()&offset.le(4320)]
        st=(ch.diagnosisstring.fillna('')+' '+ch.icd9code.fillna('')).str.lower()
        hit=st.str.contains('sepsis|septic',regex=True)
        diag_ids.update(ch.loc[hit,'patientunitstayid'].astype(int))
        boundary_ids.update(ch.loc[st.str.contains(r'\bsepsis\b|\bseptic\b',regex=True),'patientunitstayid'].astype(int))
        ambiguous_ids.update(ch.loc[hit&st.str.contains('aseptic|no sepsis|not sepsis|rule out sepsis',regex=True),'patientunitstayid'].astype(int))
        for t,n in ch.loc[hit,'diagnosisstring'].value_counts().items():terms[t]=terms.get(t,0)+int(n)
    apache=ev.apacheadmissiondx.fillna('').str.lower().str.contains('sepsis|septic',regex=True)
    text=apache|ev.patientunitstayid.isin(diag_ids)
    boundary=ev.apacheadmissiondx.fillna('').str.lower().str.contains(r'\bsepsis\b|\bseptic\b',regex=True)|ev.patientunitstayid.isin(boundary_ids)
    rsum={'alltime_original_text_sepsis':int(b(ev.sepsis_text_flag).sum()),'by72_text_sepsis':int(text.sum()),'APACHE_baseline_positive':int(apache.sum()),'diagnosis_documented_by72_positive':len(diag_ids),'diagnosis_text_potential_ambiguity_patient_stays':len(ambiguous_ids),'boundary_definition_by72_positive':int(boundary.sum()),'definition':'lowercase APACHE admission diagnosis contains sepsis|septic OR lower(diagnosisstring + space + icd9code) contains sepsis|septic in a record with observed numeric diagnosisoffset<=4320; negative offsets retained as pre-ICU documentation; no negation/adjudication; missing text empty; missing/nonnumeric offsets excluded.'}
    js(OUT/'eicu_text_algorithm_audit.json',rsum)
    pd.DataFrame([{'diagnostic_text':k,'matching_records_by72':v} for k,v in sorted(terms.items())]).to_csv(OUT/'eicu_matching_diagnostic_terms.csv',index=False)
    ev['sepsis_text_by72_flag']=text;ev['sepsis_text_boundary_by72_flag']=boundary;ev['sepsis_APACHE_flag']=apache
    return ev,text
def load_data():
    cohort=pd.read_csv(MC/'07_full_sofa_sepsis3'/'mimiciv_sepsis3_icu_admit_cohort.csv')
    outcomes=pd.read_csv(MC/'05_outcomes'/'mimiciv_outcomes.csv')
    m=cohort.merge(outcomes,on=IDS,how='inner',suffixes=('','_out'),validate='one_to_one')
    m['hospital_death']=m.hospital_expire_flag.astype(int);m['icu_death']=b(m.icu_mortality).astype(int);m['death_28d']=b(m.mortality_28d).astype(int)
    m['gender_numeric']=m.gender.eq('M').astype(int)
    wide=pd.read_csv(MC/'02_window_features'/'mimiciv_lab_window_features_0_72h_wide.csv')
    sofa=pd.read_csv(MC/'07_full_sofa_sepsis3'/'mimiciv_sepsis3_icu_admit_full_sofa_scores_0_72h.csv')
    scols=[c for c in sofa if c.startswith('sofa_')]
    m=m.merge(wide,on=IDS,validate='one_to_one').merge(pivot(sofa,IDS,scols),on=IDS,validate='one_to_one')
    m=harmonize(m)
    ml=pd.to_datetime(m.icu_intime)+pd.Timedelta(hours=72)
    hd=pd.to_datetime(m.hospital_deathtime)
    # Explicitly test recorded hospital death against landmark; date-only dod is too coarse here.
    recorded_before=hd.notna() & (hd<=ml)
    discharge_before=pd.to_datetime(m.hospital_dischtime).le(ml)
    # A date-only death strictly before the landmark calendar date contradicts survival;
    # a death date on the landmark day is not treated as a precise midnight event.
    dateonly_before=hd.isna() & pd.to_datetime(m.date_of_death).lt(ml.dt.normalize())
    early=m.icu_los_days.lt(3)
    chronology=m.icu_los_days.ge(3)&~recorded_before&~discharge_before&~dateonly_before
    pair_not_known=pd.to_datetime(m.antibiotic_time).gt(ml)|pd.to_datetime(m.culture_time).gt(ml)|m.antibiotic_time.isna()|m.culture_time.isna()
    lm=chronology&~pair_not_known
    complete=m[SIC[0]].notna()
    groups=[desc(m.loc[early&m.icu_death.eq(1)],'ICU death before 72h'),desc(m.loc[early&m.icu_death.eq(0)],'ICU discharge alive before 72h'),desc(m.loc[lm&~complete],'72h eligible; W1 SIC unavailable'),desc(m.loc[lm&complete],'Included landmark')]
    pd.DataFrame(groups).to_csv(OUT/'cohort_exclusion_characteristics_mimic.csv',index=False)
    ev=pd.read_csv(EC/'eicu_sic_trajectory_features_full_sofa.csv',low_memory=False)
    status=pd.read_csv(DATA/'eicu-collaborative-research-database-2.0'/'patient.csv.gz',usecols=['patientunitstayid','hospitaldischargestatus','unitdischargestatus'])
    ev=ev.merge(status,on='patientunitstayid',validate='one_to_one')
    # Derive missing two component series directly from retained source score file.
    es=pd.read_csv(EC/'eicu_full_sofa_sic_scores_0_72h.csv',usecols=['patientunitstayid','window_name','sofa_coagulation'])
    ev=ev.merge(pivot(es,'patientunitstayid',['sofa_coagulation']),on='patientunitstayid',validate='one_to_one')
    ev['anchor_age']=ev.age_years;ev['gender_numeric']=ev.gender.astype(str).str.lower().eq('male').astype(int)
    ev['hospital_death']=b(ev.hospital_mortality).astype(int);ev['icu_death']=b(ev.unit_mortality).astype(int)
    ev=harmonize(ev)
    ev,text=eicu_text_by_landmark(ev)
    elu=ev.unit_los_days.ge(3); el=elu & ev.hospital_los_days.gt(3); ec=ev[SIC[0]].notna()
    known_outcome=ev.hospitaldischargestatus.isin(['Alive','Expired'])
    e=ev.loc[text&el&ec&known_outcome].copy().reset_index(drop=True)
    m=m.loc[lm&complete].copy().reset_index(drop=True)
    oldm_path=ROOT/'02_可复现代码'/'output'/'full_k4'/'tables'/'cluster_assignments.csv'
    olde_path=ROOT/'02_可复现代码'/'output'/'full_k4'/'external_validation_eicu_full_sofa'/'eicu_full_sofa_external_predictions.csv'
    oldm=pd.read_csv(oldm_path,usecols=['stay_id','hospital_death']) if oldm_path.exists() else pd.DataFrame(columns=['stay_id','hospital_death'])
    olde=pd.read_csv(olde_path) if olde_path.exists() else pd.DataFrame(columns=['patientunitstayid'])
    # Aggregate event and follow-up summaries only; participant files remain local restricted access.
    report={'cohort_mimic':{'source':len(cohort),'early_total':int(early.sum()),'early_ICU_death':int((early&b(cohort.merge(outcomes,on=IDS,suffixes=('','_out')).icu_mortality)).sum()),'early_ICU_discharge_alive':groups[1]['n'],'ICU_still_present_but_recorded_hospital_death_before72':int((~early&recorded_before).sum()),'subsequent_exclusion_hospital_discharge_by72':int((~early&~recorded_before&discharge_before).sum()),'subsequent_exclusion_dateonly_death_before_landmark_date':int((~early&~recorded_before&~discharge_before&dateonly_before).sum()),'landmark_before_W1_filter':int(lm.sum()),'missing_W1_SIC':int((lm&~complete).sum()),'included':len(m),'hospital_death':int(m.hospital_death.sum()),'ICU_death':int(m.icu_death.sum()),'death_by_day28_from_ICU_admission':int(m.death_28d.sum()),'legacy_ID_set_identical':set(m.stay_id)==set(oldm.stay_id),'legacy_outcome_identical':m.set_index('stay_id').hospital_death.sort_index().equals(oldm.set_index('stay_id').hospital_death.sort_index())},'cohort_eicu':{'adult_first_ICU_stays':len(ev),'text_sepsis':int(text.sum()),'early_total':int((text&~elu).sum()),'early_ICU_death':int((text&~elu&ev.icu_death.eq(1)).sum()),'early_ICU_discharge_alive':int((text&~elu&ev.icu_death.eq(0)).sum()),'subsequent_exclusion_hospital_discharge_by72':int((text&elu&~el).sum()),'landmark_before_W1_filter':int((text&el).sum()),'missing_W1_SIC':int((text&el&~ec).sum()),'included':len(e),'hospital_death':int(e.hospital_death.sum()),'ICU_death':int(e.icu_death.sum()),'legacy_ID_set_identical':set(e.patientunitstayid)==set(olde.patientunitstayid),'Sepsis3_eligible_total':int(b(ev.sepsis3_icu_admit_flag).sum()),'Sepsis3_landmark_W1_eligible':int((b(ev.sepsis3_icu_admit_flag)&el&ec).sum())},'harmonization':{},'endpoint_definitions':{'primary':'Hospital death during index hospitalization after 72-hour landmark; variable horizon until hospital discharge; no fixed 28-day horizon.','MIMIC_28day':'Retained historic flag uses COALESCE(hospital_deathtime,date_of_death) <= ICU_intime +28days; from day3 landmark to day28 is 25 remaining days, not 28 post-landmark days.','eICU':'Hospital/unit status Expired; no post-discharge all-cause death follow-up. hospital_los_days stored in derived file is hospitaldischargeoffset/1440, measured from ICU admission, not total hospital stay.'}}
    for name,d in [('mimic',m),('eicu',e)]:
        report['harmonization'][name]={'INR_exact1_2_each_window':[int(d[f'w{i}_inr_max'].eq(1.2).sum()) for i in range(1,4)],'patients_any_INR_exact1_2':int(d[[f'w{i}_inr_max' for i in range(1,4)]].eq(1.2).any(axis=1).sum())}
        period=(pd.to_datetime(d.hospital_dischtime)-pd.to_datetime(d.icu_intime)).dt.total_seconds()/86400-3 if name=='mimic' else d.hospital_los_days-3
        report['cohort_'+name]['postlandmark_hospital_observation_days']={k:float(period.quantile(q)) for k,q in [('min',0),('q1',.25),('median',.5),('q3',.75),('max',1)]}
        report['cohort_'+name]['hospital_event_rate_pct']=float(d.hospital_death.mean()*100)
        report['cohort_'+name]['duplicate_persons']=int(d.duplicated('subject_id' if name=='mimic' else 'uniquepid').sum())
    report['cohort_mimic']['subsequent_exclusion_infection_pair_not_known_by72']=int((chronology&pair_not_known).sum())
    report['cohort_eicu']['text_sepsis_alltime_historical']=int(b(ev.sepsis_text_flag).sum())
    report['cohort_eicu']['W1_eligible_before_known_hospital_outcome_filter']=int((text&el&ec).sum())
    report['cohort_eicu']['subsequent_exclusion_unknown_hospital_outcome']=int((text&el&ec&~known_outcome).sum())
    report['cohort_eicu']['boundary_text_eligible_count']=int((ev.sepsis_text_boundary_by72_flag&el&ec&known_outcome).sum())
    report['cohort_eicu']['APACHE_text_eligible_count']=int((ev.sepsis_APACHE_flag&el&ec&known_outcome).sum())
    miss=[]
    for name,d in [('MIMIC-IV',m),('eICU',e)]:
        for c in LABS+SIC+SOFA+COMP:
            miss.append({'database':name,'feature':c,'n':len(d),'missing_n':int(d[c].isna().sum()),'missing_pct':float(d[c].isna().mean()*100)})
    pd.DataFrame(miss).to_csv(OUT/'missingness_before_carry_forward.csv',index=False)
    pd.DataFrame([desc(ev.loc[text&~elu&ev.icu_death.eq(1)],'ICU death before72',los='unit_los_days'),desc(ev.loc[text&~elu&ev.icu_death.eq(0)],'ICU discharge alive before72',los='unit_los_days'),desc(ev.loc[text&el&~ec],'72h eligible; W1 SIC unavailable',los='unit_los_days'),desc(e,'Included landmark',los='unit_los_days')]).to_csv(OUT/'cohort_exclusion_characteristics_eicu.csv',index=False)
    js(OUT/'cohort_audit.json',report)
    print(json.dumps(report,ensure_ascii=False),flush=True)
    return m,e,features(m),features(e)

class MatrixBuilder(BaseEstimator,TransformerMixin):
    def __init__(self,features,cluster_k=0): self.features=features;self.cluster_k=cluster_k
    def fit(self,X,y=None):
        self.imputer_=SimpleImputer(strategy='median',keep_empty_features=True)
        self.scaler_=StandardScaler()
        a=self.imputer_.fit_transform(X[list(self.features)])
        self.scaler_.fit(a)
        if self.cluster_k:
            self.cimputer_=SimpleImputer(strategy='median',keep_empty_features=True)
            self.cscaler_=StandardScaler()
            cx=self.cscaler_.fit_transform(self.cimputer_.fit_transform(X[CLUSTER]))
            self.kmeans_=KMeans(n_clusters=self.cluster_k,n_init=20,max_iter=500,random_state=42).fit(cx)
        return self
    def transform(self,X):
        a=self.scaler_.transform(self.imputer_.transform(X[list(self.features)]))
        if self.cluster_k:
            labs=self.kmeans_.predict(self.cscaler_.transform(self.cimputer_.transform(X[CLUSTER])))
            a=np.column_stack([a,np.eye(self.cluster_k)[labs]])
        return a

MatrixBuilder.__module__='revision_analysis'

def calibrate(y,p):
    x=logit(np.clip(p,1e-7,1-1e-7)); A=np.column_stack([np.ones(len(y)),x])
    def f(v):
        eta=A@v;pr=expit(eta)
        return np.sum(np.logaddexp(0,eta)-y*eta), A.T@(pr-y)
    fit=minimize(f,np.array([0.,1.]),jac=True,method='BFGS',options={'gtol':1e-7})
    # Calibration-in-the-large: intercept only with logit probability offset.
    a=0.
    for _ in range(20):
        q=expit(x+a);delta=np.sum(y-q)/np.sum(q*(1-q));a+=delta
        if abs(delta)<1e-9:break
    return float(fit.x[0]),float(fit.x[1]),float(a)
def metrics(y,p):
    ci,cs,citl=calibrate(y,p)
    return {'AUROC':float(roc_auc_score(y,p)),'AUPRC':float(average_precision_score(y,p)),'Brier':float(brier_score_loss(y,p)),'calibration_intercept':ci,'calibration_slope':cs,'calibration_in_the_large':citl}
def run_models(m,e):
    m1=BASE+[SOFA[0]];m2=m1+[SIC[0]];m3=m2+SUMMARY
    expanded=BASE+LABS+SIC+SOFA+COMP+[f'w{i}_sofa_4systems_for_sic' for i in range(1,4)]+SUMMARY+[f'w{i}_{v}' for v in ['sic_platelet_component','sic_inr_component','sic_organ_sofa_component'] for i in range(1,4)]
    sets={'M1_admission_SOFA':(m1,0),'M2_admission_SIC':(m2,0),'M3_continuous_SIC':(m3,0),'Coagulation_only_serial':(BASE+COAG,0),'Peak_SOFA':(BASE+['sofa_peak'],0),'Peak_SOFA_plus_SIC':(BASE+['sofa_peak',SIC[0]]+SUMMARY,0),'Serial_SOFA':(BASE+SOFA,0),'M3_plus_K2':(m3,2),'M3_plus_K4_exploratory':(m3,4),'Expanded_LR':(expanded,0),'M3_missing_indicators':(m3+[c+'_missing' for c in SIC+COAG],0),'Expanded_XGB':(expanded,0)}
    sets['M3_level_only']=(m2+SUMMARY[:-1],0)
    js(OUT/'feature_sets.json',{k:{'features':v[0],'cluster_k':v[1],'n_continuous_features':len(v[0]),'cluster_features':CLUSTER if v[1] else []} for k,v in sets.items()})
    y=m.hospital_death.to_numpy();ye=e.hospital_death.to_numpy()
    cv=StratifiedKFold(5,shuffle=True,random_state=42)
    folds=list(cv.split(m,y));oof=pd.DataFrame({'stay_id':m.stay_id,'y':y,'fold':-1});ext=pd.DataFrame({'patientunitstayid':e.patientunitstayid,'uniquepid':e.uniquepid,'hospitalid':e.hospitalid,'y':ye})
    for j,(_,va) in enumerate(folds):oof.loc[va,'fold']=j
    results=[];tuning=[];coeff=[]
    (OUT/'models').mkdir(exist_ok=True)
    for name,(cols,k) in sets.items():
        print('START',name,flush=True)
        def candidate(innerseed=42):
            if name=='Expanded_XGB':
                from xgboost import XGBClassifier
                return Pipeline([('matrix',MatrixBuilder(tuple(cols),k)),('model',XGBClassifier(n_estimators=200,max_depth=4,learning_rate=.1,subsample=.8,colsample_bytree=.8,random_state=42,eval_metric='logloss',n_jobs=2))])
            p=Pipeline([('matrix',MatrixBuilder(tuple(cols),k)),('model',LogisticRegression(max_iter=2000,solver='lbfgs',random_state=42))])
            return GridSearchCV(p,{'model__C':[.01,.1,1.,10.]},scoring='neg_log_loss',cv=StratifiedKFold(3,shuffle=True,random_state=innerseed),n_jobs=1,refit=True)
        pred=np.zeros(len(m))
        for j,(tr,va) in enumerate(folds):
            model=candidate(100+j);model.fit(m.iloc[tr],y[tr]);pred[va]=model.predict_proba(m.iloc[va])[:,1]
            tuning.append({'model':name,'fold':j,'C':model.best_params_['model__C'] if hasattr(model,'best_params_') else None})
        final=candidate();final.fit(m,y);ep=final.predict_proba(e)[:,1]
        pipe=final.best_estimator_ if hasattr(final,'best_estimator_') else final
        tuning.append({'model':name,'fold':'full_derivation','C':final.best_params_['model__C'] if hasattr(final,'best_params_') else None})
        joblib.dump(pipe,OUT/'models'/f'{name}.joblib')
        if hasattr(pipe.named_steps['model'],'coef_'):
            coef=pipe.named_steps['model'].coef_[0];names=cols+[f'cluster_{c}' for c in range(k)]
            coeff.append({'model':name,'feature':'INTERCEPT','coefficient':float(pipe.named_steps['model'].intercept_[0])})
            for c,v in zip(names,coef):coeff.append({'model':name,'feature':c,'coefficient':float(v)})
            js(OUT/'models'/f'{name}_preprocessing.json',{'features':cols,'median':pipe.named_steps['matrix'].imputer_.statistics_.tolist(),'mean':pipe.named_steps['matrix'].scaler_.mean_.tolist(),'scale':pipe.named_steps['matrix'].scaler_.scale_.tolist(),'C':pipe.named_steps['model'].C})
        oof[name]=pred;ext[name]=ep
        for cohort,yy,pp in [('MIMIC-IV',y,pred),('eICU',ye,ep)]:results.append({'model':name,'cohort':cohort,'n':len(yy),'events':int(yy.sum()),**metrics(yy,pp)})
        print(name,json.dumps(results[-2:]),flush=True)
        oof.to_csv(OUT/'internal_predictions_RESTRICTED.csv',index=False);ext.to_csv(OUT/'external_predictions_RESTRICTED.csv',index=False)
        pd.DataFrame(results).to_csv(OUT/'performance_point_estimates.csv',index=False)
        pd.DataFrame(tuning).to_csv(OUT/'tuning.csv',index=False)
        pd.DataFrame(coeff).to_csv(OUT/'logistic_coefficients_standardized.csv',index=False)
    return oof,ext,sets

def bootstrap(oof,ext,sets,B=1000,only_external=False):
    rows=[];differences=[];curve=[]
    if only_external:
        rows=pd.read_csv(OUT/'performance_with_95CI.csv').query("cohort == 'MIMIC-IV'").to_dict('records')
        differences=pd.read_csv(OUT/'paired_differences_with_95CI.csv').query("cohort == 'MIMIC-IV'").to_dict('records')
        curve=pd.read_csv(OUT/'calibration_deciles.csv').query("cohort == 'MIMIC-IV'").to_dict('records')
    pairs=[('M2_admission_SIC','M1_admission_SOFA'),('M3_continuous_SIC','M2_admission_SIC'),('M3_continuous_SIC','Peak_SOFA'),('Peak_SOFA_plus_SIC','Peak_SOFA'),('Coagulation_only_serial','M3_continuous_SIC'),('Serial_SOFA','M3_continuous_SIC'),('M3_plus_K2','M3_continuous_SIC'),('M3_plus_K4_exploratory','M3_continuous_SIC'),('Expanded_LR','M3_continuous_SIC'),('Expanded_XGB','Expanded_LR'),('M3_missing_indicators','M3_continuous_SIC'),('M3_continuous_SIC','M3_level_only')]
    for cohort,d in [('MIMIC-IV',oof),('eICU',ext)]:
        if only_external and cohort=='MIMIC-IV':continue
        print('BOOTSTRAP',cohort,flush=True)
        rng=np.random.default_rng(42);yy=d.y.to_numpy();n=len(d)
        # Cluster bootstrap eICU by person to account for repeat hospitalizations.
        groups=[np.array([i]) for i in range(n)] if cohort=='MIMIC-IV' else list(d.groupby('uniquepid',sort=False).indices.values())
        bs={name:[] for name in sets}
        for bidx in range(B):
            ii=np.concatenate([groups[g] for g in rng.integers(0,len(groups),len(groups))])
            for name in sets:bs[name].append(metrics(yy[ii],d[name].to_numpy()[ii]))
            if bidx%100==0: print(cohort,bidx,flush=True)
        for name in sets:
            pt=metrics(yy,d[name].to_numpy());bdf=pd.DataFrame(bs[name])
            row={'cohort':cohort,'model':name,'n':n,'events':int(yy.sum())}
            for metric,val in pt.items():row[metric]=val;row[metric+'_lo']=float(bdf[metric].quantile(.025));row[metric+'_hi']=float(bdf[metric].quantile(.975))
            rows.append(row)
            pp=d[name].to_numpy();bins=pd.qcut(pp,10,duplicates='drop')
            dd=pd.DataFrame({'p':pp,'y':yy,'bin':bins})
            for _,q in dd.groupby('bin',observed=True):
                num=len(q);rate=q.y.mean();den=1+1.96**2/num;center=(rate+1.96**2/(2*num))/den;half=1.96*np.sqrt(rate*(1-rate)/num+1.96**2/(4*num*num))/den
                curve.append({'cohort':cohort,'model':name,'n':num,'mean_predicted':q.p.mean(),'observed':rate,'observed_lo':center-half,'observed_hi':center+half})
        for a,bname in pairs:
            for metric in ['AUROC','AUPRC','Brier']:
                vals=np.array([v[metric] for v in bs[a]])-np.array([v[metric] for v in bs[bname]])
                differences.append({'cohort':cohort,'model_a':a,'model_b':bname,'metric':metric,'difference':metrics(yy,d[a].to_numpy())[metric]-metrics(yy,d[bname].to_numpy())[metric],'lo':np.quantile(vals,.025),'hi':np.quantile(vals,.975)})
        pd.DataFrame(rows).to_csv(OUT/'performance_with_95CI.csv',index=False)
        pd.DataFrame(differences).to_csv(OUT/'paired_differences_with_95CI.csv',index=False)
        pd.DataFrame(curve).to_csv(OUT/'calibration_deciles.csv',index=False)
    return rows

def run():
    with threadpool_limits(limits=2):
        rawm,rawe,m,e=load_data()
        if '--audit-only' in sys.argv:return
        oof,ext,sets=run_models(m,e)
        bootstrap(oof,ext,sets)
    import sklearn,scipy
    js(OUT/'environment.json',{'Python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scikit_learn':sklearn.__version__,'scipy':scipy.__version__,'random_seed':42,'bootstrap_replicates':1000,'internal_validation':'5 outer x3 inner stratified CV; inner C selection by log-loss; fixed feature sets; all fitted preprocessing inside inner/outer training data','CI_limitation':'percentile bootstrap of fixed cross-fitted/frozen predictions; MIMIC person-level, eICU person-cluster resampling; intervals do not include pipeline refitting variability or eICU between-hospital sampling variability','new_analysis':'Reviewer-requested post hoc reanalysis, not prespecified.'})
if __name__=='__main__':run()
