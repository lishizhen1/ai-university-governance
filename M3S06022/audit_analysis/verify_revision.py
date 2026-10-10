"""Verify frozen model outputs and finalized cohort definitions without refitting."""
import json,hashlib,sys
import revision_analysis as r
import numpy as np,pandas as pd,joblib
from threadpoolctl import threadpool_limits

def run():
    with threadpool_limits(limits=2):
        rawm,rawe,m,e=r.load_data();out=r.OUT
        ext=pd.read_csv(out/'external_predictions_RESTRICTED.csv');oof=pd.read_csv(out/'internal_predictions_RESTRICTED.csv')
        assert len(m)==len(oof) and len(e)==len(ext)
        assert np.array_equal(m.stay_id,oof.stay_id) and np.array_equal(e.patientunitstayid,ext.patientunitstayid)
        assert np.array_equal(m.hospital_death,oof.y) and np.array_equal(e.hospital_death,ext.y)
        names=json.loads((out/'feature_sets.json').read_text())
        checks=[]
        for name in names:
            model=joblib.load(out/'models'/f'{name}.joblib')
            pred=model.predict_proba(e)[:,1]
            err=float(np.max(np.abs(pred-ext[name].to_numpy())))
            assert err<1e-7,(name,err)
            assert np.isfinite(oof[name]).all() and np.isfinite(pred).all()
            checks.append({'model':name,'maximum_prediction_reload_difference':err,'passed':True})
        assert not (pd.to_datetime(m.antibiotic_time)>pd.to_datetime(m.icu_intime)+pd.Timedelta(hours=72)).any()
        assert not (pd.to_datetime(m.culture_time)>pd.to_datetime(m.icu_intime)+pd.Timedelta(hours=72)).any()
        assert e.sepsis_text_by72_flag.all()
        statuses=pd.read_csv(r.DATA/'eicu-collaborative-research-database-2.0'/'patient.csv.gz',usecols=['patientunitstayid','hospitaldischargestatus','unitdischargestatus','unitvisitnumber'])
        statuses=statuses.set_index('patientunitstayid').loc[e.patientunitstayid]
        sanity={'MIMIC_unit_death_but_hospital_survived':int((m.icu_death.eq(1)&m.hospital_death.eq(0)).sum()),'eICU_unit_death_but_hospital_survived':int((e.icu_death.eq(1)&e.hospital_death.eq(0)).sum()),'eICU_unique_people':int(e.uniquepid.nunique()),'eICU_hospitals':int(e.hospitalid.nunique()),'eICU_unknown_hospital_discharge_status':int((~statuses.hospitaldischargestatus.isin(['Alive','Expired'])).sum()),'eICU_unknown_unit_discharge_status':int((~statuses.unitdischargestatus.isin(['Alive','Expired'])).sum()),'MIMIC_age_range':[float(m.anchor_age.min()),float(m.anchor_age.max())],'eICU_age_range':[float(e.anchor_age.min()),float(e.anchor_age.max())]}
        sanity['MIMIC_nonbinary_or_missing_sex']=int((~m.gender.isin(['M','F'])).sum())
        sanity['eICU_nonbinary_or_missing_sex']=int((~e.gender.str.lower().isin(['male','female'])).sum())
        sanity['MIMIC_hospital_outcome_not_0_or_1']=int((~m.hospital_death.isin([0,1])).sum())
        r.js(out/'verification.json',{'cohort_alignment':True,'landmark_information_timing':True,'models':checks,'status_sanity':sanity})
        print(json.dumps(sanity),flush=True)
if __name__=='__main__':run()
