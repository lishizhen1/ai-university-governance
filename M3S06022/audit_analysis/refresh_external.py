"""Rescore frozen derivation models after excluding missing external outcomes."""
import revision_analysis as r
import numpy as np,pandas as pd,joblib,json,platform
from threadpoolctl import threadpool_limits
def run():
    with threadpool_limits(limits=2):
        _,_,m,e=r.load_data()
        sets=json.loads((r.OUT/'feature_sets.json').read_text())
        ext=pd.DataFrame({'patientunitstayid':e.patientunitstayid,'uniquepid':e.uniquepid,'hospitalid':e.hospitalid,'y':e.hospital_death})
        results=pd.read_csv(r.OUT/'performance_point_estimates.csv').query("cohort == 'MIMIC-IV'").to_dict('records')
        for name in sets:
            mdl=joblib.load(r.OUT/'models'/f'{name}.joblib');p=mdl.predict_proba(e)[:,1];ext[name]=p
            results.append({'model':name,'cohort':'eICU','n':len(e),'events':int(e.hospital_death.sum()),**r.metrics(e.hospital_death.to_numpy(),p)})
        ext.to_csv(r.OUT/'external_predictions_RESTRICTED.csv',index=False)
        pd.DataFrame(results).to_csv(r.OUT/'performance_point_estimates.csv',index=False)
        print('FROZEN_EXTERNAL_REFRESHED',len(e),flush=True)
        oof=pd.read_csv(r.OUT/'internal_predictions_RESTRICTED.csv')
        r.bootstrap(oof,ext,sets,only_external=True)
        import sklearn,scipy,xgboost
        r.js(r.OUT/'environment.json',{'Python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scikit_learn':sklearn.__version__,'scipy':scipy.__version__,'xgboost':xgboost.__version__,'random_seed':42,'bootstrap_replicates':1000,'internal_validation':'5 outer x3 inner stratified CV; inner C selection by log-loss; fixed feature sets; all fitted preprocessing inside inner/outer training data','CI_limitation':'percentile bootstrap of fixed cross-fitted/frozen predictions; MIMIC person-level, eICU person-cluster resampling; intervals do not include pipeline refitting variability or eICU between-hospital sampling variability','new_analysis':'Reviewer-requested post hoc reanalysis, external cohort previously examined; no external tuning.'})
if __name__=='__main__':run()
