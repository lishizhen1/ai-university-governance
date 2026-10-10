"""Validate and hash the finalized principal-analysis snapshot."""
from pathlib import Path
import json,hashlib
import pandas as pd
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run():
    co=json.loads((HERE/'cohort_audit.json').read_text(encoding='utf-8'))
    assert co['cohort_mimic']['included']==10810 and co['cohort_mimic']['hospital_death']==2174
    assert co['cohort_eicu']['included']==4355 and co['cohort_eicu']['hospital_death']==979
    perf=pd.read_csv(HERE/'performance_with_95CI.csv')
    assert len(perf)==26
    for cohort,n,event in [('MIMIC-IV',10810,2174),('eICU',4355,979)]:
        s=perf[perf.cohort.eq(cohort)];assert len(s)==13 and s.n.eq(n).all() and s.events.eq(event).all()
    prof=pd.read_csv(HERE/'cluster_profiles.csv')
    for (cohort,k),g in prof.groupby(['cohort','k']):assert g.n.sum()==(10810 if cohort=='MIMIC-IV' else 4355)
    for name,n in [('internal_predictions_RESTRICTED.csv',10810),('external_predictions_RESTRICTED.csv',4355)]:assert len(pd.read_csv(HERE/name))==n
    missing=pd.read_csv(HERE/'missingness_before_carry_forward.csv');assert missing.loc[missing.database.eq('eICU'),'n'].eq(4355).all()
    stability=pd.read_csv(HERE/'cluster_stability_distribution.csv');assert stability.groupby('k').size().eq(200).all()
    verify=json.loads((HERE/'verification.json').read_text());assert all(r['passed'] for r in verify['models'])
    files=[p for p in HERE.iterdir() if p.is_file() and p.suffix in ['.csv','.json','.md','.py','.txt'] and p.name!='FINAL_SNAPSHOT.json']
    files+=list((HERE/'models').glob('*'))
    payload={'snapshot':'revision_main_v3_final','status':'FROZEN_FOR_MANUSCRIPT_ASSEMBLY','cohorts':{'MIMIC-IV':{'n':10810,'hospital_events':2174},'eICU':{'n':4355,'hospital_events':979}},'predicates':{'MIMIC-IV':'Original adult first-ICU operational Sepsis3 source; ICU_LOS>=3; no recorded hospital death/discharge<=72h; no date-only death before landmark calendar date; both infection-pair timestamps<=72h; observedW1platelet+INR','eICU':'Original adult first-unit-per-hospitalization source; APACHE admission regex OR observed numeric diagnosisoffset<=4320 regex; ICU_LOS>=3; hospitaldischargeoffset>4320; observedW1platelet+INR; hospitaldischargestatus Alive or Expired'},'intervals':{'performance_bootstrap':1000,'cluster_bootstrap':200,'limitation':'Performance intervals condition on fixed cross-fitted/frozen predictions; no pipeline refitting bootstrap.'},'sha256':{str(p.relative_to(HERE)):sha(p) for p in sorted(files) if p.is_file()},'restricted_outputs':[str(p.relative_to(HERE)) for p in files if '_RESTRICTED' in p.name],'superseded':'legacy_before_time_ascertainment excluded; earlier logs may mention intermediate counts. Do not assemble figures/tables from legacy snapshots.'}
    (HERE/'FINAL_SNAPSHOT.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf-8')
    print('FROZEN',payload['cohorts'],len(payload['sha256']),'files')
if __name__=='__main__':run()
