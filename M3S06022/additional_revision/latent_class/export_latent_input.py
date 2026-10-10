"""Create restricted local longitudinal inputs without outcomes for lcmm sensitivity."""
from pathlib import Path
import contextlib
import json
import sys
import hashlib

HERE=Path(__file__).resolve().parent
AUDIT=HERE.parents[1]/'audit_analysis'
sys.path.insert(0,str(AUDIT))
import revision_analysis as r

PRIVATE=HERE/'private_local_only'
PRIVATE.mkdir(exist_ok=True)
r.OUT=PRIVATE/'cohort_load_audit'
r.OUT.mkdir(exist_ok=True)
with (PRIVATE/'cohort_load.log').open('w',encoding='utf-8') as log, contextlib.redirect_stdout(log):
    rawm,rawe,m,e=r.load_data()
assert len(rawm)==10810 and len(rawe)==4355
meta={'cohort_source':'Frozen revision_analysis.load_data, harmonized SIC and final eligibility',
      'no_outcome_columns_exported':True,'time_unit':'24-hour window step; W1=0,W2=1,W3=2',
      'missingness':'Only observed pre-carry-forward SIC values contribute; no carried or imputed SIC is used.',
      'privacy':'Inputs, subject posteriors and fitted models remain restricted local files; no publication is authorized.'}
for label,raw,person in [('mimic',rawm,'subject_id'),('eicu',rawe,'uniquepid')]:
    wide=raw[r.SIC].copy()
    wide.columns=['W1','W2','W3']
    wide.insert(0,'id',range(1,len(wide)+1))
    long=wide.melt(id_vars='id',var_name='window',value_name='sic')
    long['time']=long.window.map({'W1':0,'W2':1,'W3':2})
    long=long.dropna(subset=['sic']).sort_values(['id','time'])[['id','time','sic']]
    assert long.sic.between(0,6).all() and (long.sic==long.sic.round()).all()
    path=PRIVATE/f'{label}_observed_long.csv'
    long.to_csv(path,index=False)
    complete_ids=wide.loc[wide[['W1','W2','W3']].notna().all(axis=1),'id']
    long.loc[long.id.isin(complete_ids)].to_csv(PRIVATE/f'{label}_complete_long.csv',index=False)
    # Mapping is retained only for potential local checks; never needed to fit or select class counts.
    r.pd.DataFrame({'id':wide.id,'source_person':raw[person].astype(str).values}).to_csv(PRIVATE/f'{label}_person_mapping.csv',index=False)
    counts=long.groupby('id').size().value_counts().sort_index()
    meta[label]={'subjects':len(wide),'observed_measurements':len(long),'complete_subjects':len(complete_ids),
                 'measurement_count_distribution':{str(k):int(v) for k,v in counts.items()},
                 'unique_source_persons':int(raw[person].nunique()),
                 'by_time':long.groupby('time').sic.agg(['count','mean','std']).reset_index().to_dict('records'),
                 'score_counts':long.sic.value_counts().sort_index().astype(int).to_dict(),
                 'input_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
(HERE/'input_metadata.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(meta,indent=2,ensure_ascii=True))
