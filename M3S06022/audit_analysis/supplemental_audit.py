"""Post-revision cluster diagnostics, transport checks, and transparency outputs."""
from pathlib import Path
import json,sys,re,hashlib,os
import revision_analysis as r
import numpy as np,pandas as pd
from sklearn.metrics import silhouette_score, adjusted_rand_score, silhouette_samples
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.special import expit,logit
from threadpoolctl import threadpool_limits
import joblib
MatrixBuilder=r.MatrixBuilder
OUT=r.OUT

def source_definition_root():
    """Resolve immutable original definitions in either workspace or delivery layout."""
    env=os.environ.get('SIC_SOURCE_CODE_ROOT')
    candidates=([Path(env)] if env else [])+[
        OUT/'source_definitions',OUT/'source_definitions'/'02_可复现代码',
        r.ROOT/'02_可复现代码',
    ]
    for p in candidates:
        if (p/'01_数据预处理'/'process_sic_trajectory_data.py').exists():return p
    raise FileNotFoundError('Original source definitions not found. Include source_definitions/01_数据预处理 etc. alongside this script, or set SIC_SOURCE_CODE_ROOT to the original 02_可复现代码 directory.')

def cluster_diagnostics(m,e,B=200):
    summaries=[];profiles=[];centroidrows=[];stabilities=[];transport=[];assignment=[];profile_boot=[]
    fit=r.MatrixBuilder(tuple(r.CLUSTER),0).fit(m)
    x=fit.transform(m);xe=fit.transform(e)
    for k in range(2,7):
        km=KMeans(n_clusters=k,n_init=20,max_iter=500,random_state=42).fit(x)
        label=km.labels_;sizes=np.bincount(label)
        sil=silhouette_score(x,label)
        summaries.append({'k':k,'silhouette':sil,'min_cluster_pct':sizes.min()/len(m)*100,'sizes':sizes.tolist()})
        print('CLUSTER',k,'silhouette',sil,flush=True)
        if k not in [2,4]:continue
        em=km.predict(xe)
        orig=fit.scaler_.inverse_transform(km.cluster_centers_)
        rank=np.argsort(orig[:,:3].mean(axis=1));name={int(c):f'K{k} stratum {j+1}' for j,c in enumerate(rank)}
        dx=km.transform(x);de=km.transform(xe)
        margin=(np.sort(dx,axis=1)[:,1]-np.min(dx,axis=1))/np.sort(dx,axis=1)[:,1]
        silhouette=silhouette_samples(x,label)
        assignment.append({'k':k,'mean_silhouette':float(silhouette.mean()),'negative_silhouette_pct':float((silhouette<0).mean()*100),'nearest_distance_margin_median':float(np.median(margin)),'nearest_distance_margin_q1':float(np.quantile(margin,.25)),'nearest_distance_margin_q3':float(np.quantile(margin,.75)),'margin_below_0_1_pct':float((margin<.1).mean()*100)})
        for c in range(k):
            for cohort,df,labels,dist in [('MIMIC-IV',m,label,dx),('eICU',e,em,de)]:
                mask=labels==c
                profiles.append({'k':k,'cohort':cohort,'cluster':c,'stratum':name[c],'n':int(mask.sum()),'hospital_deaths':int(df.loc[mask,'hospital_death'].sum()),'hospital_mortality_pct':float(df.loc[mask,'hospital_death'].mean()*100),**{f'w{j+1}_SIC_mean':float(df.loc[mask,sc].mean()) for j,sc in enumerate(r.SIC)},'W1_SOFA_mean':float(df.loc[mask,r.SOFA[0]].mean())})
                nearest=dist[mask,c]
                transport.append({'k':k,'cohort':cohort,'stratum':name[c],'n':int(mask.sum()),'distance_median':float(np.median(nearest)),'distance_q1':float(np.quantile(nearest,.25)),'distance_q3':float(np.quantile(nearest,.75)),'distance_p95':float(np.quantile(nearest,.95)),'above_derivation_cluster_p95_pct':float((nearest>np.quantile(dx[label==c,c],.95)).mean()*100)})
        rng=np.random.default_rng(42);centers=[];agreement=np.zeros(len(m));size_b=[]
        for bi in range(B):
            ix=rng.integers(0,len(m),len(m));bm=m.iloc[ix]
            prep=r.MatrixBuilder(tuple(r.CLUSTER),0).fit(bm);bx=prep.transform(bm)
            bk=KMeans(n_clusters=k,n_init=20,max_iter=500,random_state=42+bi).fit(bx)
            bc=prep.scaler_.inverse_transform(bk.cluster_centers_)
            # Align boot centroids in reference standardized coordinates; no outcome used.
            ri,ci=linear_sum_assignment(cdist(km.cluster_centers_,fit.scaler_.transform(bc)))
            aligned=bc[ci[np.argsort(ri)]];centers.append(aligned)
            mapping={int(bb):int(rr) for rr,bb in zip(ri,ci)}
            bp=np.array([mapping[int(c)] for c in bk.predict(prep.transform(m))])
            agreement+=(bp==label)
            stabilities.append({'k':k,'bootstrap':bi,'ARI':float(adjusted_rand_score(label,bp))})
            for c in range(k):
                # Bootstrap mean of carried-forward SIC in reference stratum (sampling CI).
                means=bm.loc[label[ix]==c,r.SIC].mean().tolist()
                profile_boot.append({'k':k,'cluster':c,'bootstrap':bi,'w1':means[0],'w2':means[1],'w3':means[2]})
            if bi%25==0:print('BOOT CLUSTER',k,bi,flush=True)
        centers=np.array(centers)
        for c in range(k):
            for j,f in enumerate(r.CLUSTER):centroidrows.append({'k':k,'cluster':c,'stratum':name[c],'feature':f,'centroid_original_units':float(orig[c,j]),'centroid_standardized':float(km.cluster_centers_[c,j]),'bootstrap_lo':float(np.quantile(centers[:,c,j],.025)),'bootstrap_hi':float(np.quantile(centers[:,c,j],.975))})
        assignment[-1]['bootstrap_assignment_agreement_median']=float(np.median(agreement/B));assignment[-1]['bootstrap_assignment_agreement_q1']=float(np.quantile(agreement/B,.25));assignment[-1]['bootstrap_assignment_agreement_below80pct']=float((agreement/B<.8).mean()*100)
        pd.DataFrame({'stay_id':m.stay_id,'k':k,'cluster':label,'stratum':[name[int(c)] for c in label],'assignment_margin':margin,'bootstrap_assignment_agreement':agreement/B,**{s:m[s] for s in r.SIC}}).to_csv(OUT/f'cluster_assignments_K{k}_RESTRICTED.csv',index=False)
        pd.DataFrame({'patientunitstayid':e.patientunitstayid,'k':k,'cluster':em,'stratum':[name[int(c)] for c in em]}).to_csv(OUT/f'external_cluster_assignments_K{k}_RESTRICTED.csv',index=False)
        joblib.dump({'preprocessor':fit,'kmeans':km,'names':name},OUT/'models'/f'clustering_K{k}.joblib')
        r.js(OUT/'models'/f'clustering_K{k}_preprocessing.json',{'features':r.CLUSTER,'median':fit.imputer_.statistics_.tolist(),'mean':fit.scaler_.mean_.tolist(),'scale':fit.scaler_.scale_.tolist(),'names':name})
        pd.DataFrame(profiles).to_csv(OUT/'cluster_profiles.csv',index=False)
        pd.DataFrame(centroidrows).to_csv(OUT/'centroids_with_bootstrap_CI.csv',index=False)
        pd.DataFrame(stabilities).to_csv(OUT/'cluster_stability_distribution.csv',index=False)
        pd.DataFrame(transport).to_csv(OUT/'cluster_transport_distances.csv',index=False)
        pd.DataFrame(assignment).to_csv(OUT/'cluster_assignment_confidence.csv',index=False)
        pd.DataFrame(profile_boot).to_csv(OUT/'cluster_profile_bootstrap.csv',index=False)
    r.js(OUT/'cluster_selection.json',summaries)

def extra_external(e):
    ext=pd.read_csv(OUT/'external_predictions_RESTRICTED.csv')
    e=e.set_index('patientunitstayid').loc[ext.patientunitstayid].reset_index()
    apache=e.apacheadmissiondx.fillna('').str.lower().str.contains(r'\bseps(?:is)?\b|\bseptic\b',regex=True)
    # Exact original APACHE substring sensitivity used for comparability.
    apache=e.apacheadmissiondx.fillna('').str.lower().str.contains('sepsis|septic',regex=True)
    rows=[]
    for name in [c for c in ext if c not in ['patientunitstayid','uniquepid','hospitalid','y']]:
        p=ext[name].to_numpy();y=ext.y.to_numpy()
        rows.append({'analysis':'APACHE admission text positive subset (no refitting)','model':name,'n':int(apache.sum()),'events':int(y[apache].sum()),**r.metrics(y[apache],p[apache])})
    pd.DataFrame(rows).to_csv(OUT/'external_APACHE_subset.csv',index=False)
    rec=[];coeff=[]
    for name in ['M3_continuous_SIC','Expanded_LR','Expanded_XGB']:
        p=ext[name].to_numpy();y=ext.y.to_numpy();x=logit(np.clip(p,1e-7,1-1e-7)).reshape(-1,1)
        cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42);pred=np.zeros(len(e))
        for fold,(tr,va) in enumerate(cv.split(x,y,e.uniquepid)):
            mdl=LogisticRegression(C=1e6,max_iter=2000).fit(x[tr],y[tr]);pred[va]=mdl.predict_proba(x[va])[:,1]
            coeff.append({'model':name,'fold':fold,'intercept':float(mdl.intercept_[0]),'slope':float(mdl.coef_[0,0])})
        a,s,citl=r.calibrate(y,p)
        coeff.append({'model':name,'fold':'full_external_apparent','intercept':a,'slope':s})
        for kind,pv in [('original_frozen',p),('external_5fold_group_recalibration',pred),('apparent_external_recalibration',expit(a+s*x.ravel()))]:
            rec.append({'model':name,'analysis':kind,'n':len(y),'events':int(y.sum()),**r.metrics(y,pv)})
        ext[name+'_recalibrated_crossfit']=pred
    pd.DataFrame(rec).to_csv(OUT/'external_recalibration.csv',index=False);pd.DataFrame(coeff).to_csv(OUT/'external_recalibration_coefficients.csv',index=False)
    ext.to_csv(OUT/'external_recalibration_predictions_RESTRICTED.csv',index=False)

def transparency(m,e,rawm=None,rawe=None):
    src=source_definition_root()
    sources=[src/'01_数据预处理'/'process_sic_trajectory_data.py',src/'02_SOFA_SIC派生'/'derive_full_sofa_sepsis3_mimiciv.py',src/'02_SOFA_SIC派生'/'derive_full_sofa_sepsis3_eicu.py',src/'03_轨迹聚类与预测模型'/'run_sic_trajectory_analysis.py',src/'04_外部验证'/'run_eicu_full_external_validation.py']
    meta={'source_definition_root':str(src),'source_code_sha256':{str(p.relative_to(src)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},'original_frozen_objects_found':list(str(p) for p in src.rglob('*.joblib'))+list(str(p) for p in src.rglob('*.pkl')),'new_objects':list(p.name for p in (OUT/'models').glob('*.joblib')),'analysis_status':'Reviewer-requested post hoc reanalysis; external cohort already previously inspected, no external outcome in model fitting or hyperparameter selection. No preregistration/protocol timestamp located. No permanent repository/PID provided. Restricted data not authorized for redistribution.'}
    r.js(OUT/'provenance.json',meta)
    feats=json.loads((OUT/'feature_sets.json').read_text())
    rows=[]
    allf=sorted({f for s in feats.values() for f in s['features']}|set(r.CLUSTER))
    for f in allf:
        unit='dimensionless score'
        if 'platelet_min' in f:unit='10^9/L'
        if 'inr_max' in f:unit='ratio'
        if any(s in f for s in ['bilirubin_total_max','creatinine_max']):unit='mg/dL'
        if 'lactate_max' in f:unit='mmol/L'
        if f=='anchor_age':unit='years (MIMIC anchor_age; eICU >89 coded90)'
        if f=='gender_numeric':unit='1 male,0 otherwise as recorded'
        if f.endswith('_missing'):unit='1 unobserved before carry-forward,0 observed'
        if f=='sic_slope':unit='SIC points per24h; (W3−W1)/2'
        rows.append({'feature':f,'unit':unit,'window':{'w1':'0<=hours<24','w2':'24<=hours<48','w3':'48<=hours<72'}.get(f[:2],'summary over0–72h or baseline'),'models':'; '.join(k for k,v in feats.items() if f in v['features'])})
    pd.DataFrame(rows).to_csv(OUT/'feature_dictionary.csv',index=False)
    sub=[]
    for cohort,d in [('MIMIC-IV',m),('eICU',e)]:
        for label,col,th in [('age','anchor_age',65),('W1_SOFA',r.SOFA[0],8),('W1_SIC',r.SIC[0],4),('W1_renal_SOFA','w1_sofa_renal',2),('W1_cardiovascular_SOFA','w1_sofa_cardiovascular',2),('W1_respiratory_SOFA','w1_sofa_respiration',2)]:
            for above in [False,True]:
                mask=d[col].ge(th) if above else d[col].lt(th)
                sub.append({'database':cohort,'subgroup':label+('>=' if above else '<')+str(th),'n':int(mask.sum()),'hospital_events':int(d.loc[mask,'hospital_death'].sum())})
        for sex in [0,1]:
            mask=d.gender_numeric.eq(sex);sub.append({'database':cohort,'subgroup':'male' if sex else 'female/other','n':int(mask.sum()),'hospital_events':int(d.loc[mask,'hospital_death'].sum())})
        if cohort=='MIMIC-IV':
            for col in ['sepsis_icd_flag','admission_type','first_careunit']:
                for val,g in d.groupby(col,dropna=False):sub.append({'database':cohort,'subgroup':col+'='+str(val),'n':len(g),'hospital_events':int(g.hospital_death.sum())})
    pd.DataFrame(sub).to_csv(OUT/'subgroup_event_counts.csv',index=False)
    patterns=[]
    for cohort,d in [('MIMIC-IV',rawm),('eICU',rawe)]:
        if d is None:continue
        for v in ['sic_total_score','platelet_min','inr_max']:
            pat=d[[f'w{i}_{v}' for i in range(1,4)]].notna().astype(int).astype(str).agg(''.join,axis=1)
            for vv,ii in pat.groupby(pat).groups.items():patterns.append({'database':cohort,'variable':v,'observed_pattern_W1_W2_W3':vv,'n':len(ii),'pct':len(ii)/len(d)*100,'hospital_events':int(d.loc[ii,'hospital_death'].sum())})
    pd.DataFrame(patterns).to_csv(OUT/'joint_missingness_patterns.csv',index=False)

def run():
    with threadpool_limits(limits=2):
        rawm,rawe,m,e=r.load_data()
        # Normalize pickles initially emitted from an executed __main__ module.
        for p in (OUT/'models').glob('*.joblib'):
            obj=joblib.load(p);joblib.dump(obj,p)
        transparency(m,e,rawm,rawe);extra_external(e)
        if '--external-only' in sys.argv:
            # MIMIC fits and clustering stay unchanged after excluding unknown eICU outcomes.
            profiles=pd.read_csv(OUT/'cluster_profiles.csv').query("cohort == 'MIMIC-IV'").to_dict('records')
            transport=pd.read_csv(OUT/'cluster_transport_distances.csv').query("cohort == 'MIMIC-IV'").to_dict('records')
            for k in [2,4]:
                obj=joblib.load(OUT/'models'/f'clustering_K{k}.joblib');prep=obj['preprocessor'];km=obj['kmeans'];names=obj['names']
                x=prep.transform(m);xe=prep.transform(e);ml=km.predict(x);el=km.predict(xe);md=km.transform(x);ed=km.transform(xe)
                for c in range(k):
                    mask=el==c;nearest=ed[mask,c]
                    profiles.append({'k':k,'cohort':'eICU','cluster':c,'stratum':names[c],'n':int(mask.sum()),'hospital_deaths':int(e.loc[mask,'hospital_death'].sum()),'hospital_mortality_pct':float(e.loc[mask,'hospital_death'].mean()*100),**{f'w{j+1}_SIC_mean':float(e.loc[mask,sc].mean()) for j,sc in enumerate(r.SIC)},'W1_SOFA_mean':float(e.loc[mask,r.SOFA[0]].mean())})
                    transport.append({'k':k,'cohort':'eICU','stratum':names[c],'n':int(mask.sum()),'distance_median':float(np.median(nearest)),'distance_q1':float(np.quantile(nearest,.25)),'distance_q3':float(np.quantile(nearest,.75)),'distance_p95':float(np.quantile(nearest,.95)),'above_derivation_cluster_p95_pct':float((nearest>np.quantile(md[ml==c,c],.95)).mean()*100)})
                pd.DataFrame({'patientunitstayid':e.patientunitstayid,'k':k,'cluster':el,'stratum':[names[int(c)] for c in el]}).to_csv(OUT/f'external_cluster_assignments_K{k}_RESTRICTED.csv',index=False)
            pd.DataFrame(profiles).to_csv(OUT/'cluster_profiles.csv',index=False);pd.DataFrame(transport).to_csv(OUT/'cluster_transport_distances.csv',index=False)
        else:cluster_diagnostics(m,e)
if __name__=='__main__':run()
