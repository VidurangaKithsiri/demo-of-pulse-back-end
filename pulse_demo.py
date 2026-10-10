"""Synthetic-only pulse-pattern demonstration. No clinical diagnoses."""
import argparse
import csv
import json
import pickle
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks, welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import sklearn

ROOT = Path(__file__).resolve().parent
FS = 100
FEATURES = ['bpm_estimate', 'interval_cv', 'interval_range', 'spectral_concentration', 'high_frequency_fraction', 'flat_fraction']
KINDS = ['regular', 'slow', 'fast', 'irregular', 'noise', 'flat', 'dropout']

def generate(kind, rng):
    t = np.arange(3000) / FS
    bpm = rng.uniform(65, 95)
    if kind == 'slow': bpm = rng.uniform(38, 52)
    if kind == 'fast': bpm = rng.uniform(115, 155)
    x = np.zeros_like(t)
    beat = -1.
    width = rng.uniform(.045, .085)
    while beat < 31:
        interval = 60 / bpm * (rng.uniform(.50, 1.55) if kind == 'irregular' else rng.uniform(.975, 1.025))
        beat += interval
        x += np.exp(-.5 * ((t-beat)/width)**2)
        x += .22 * np.exp(-.5 * ((t-beat-.20)/(width*1.5))**2)
    x += .1*np.sin(2*np.pi*rng.uniform(.1,.3)*t) + rng.normal(0,.025,len(t))
    if kind == 'noise': x += rng.normal(0, rng.uniform(1,2),len(t))
    if kind == 'flat': x[:] = 0
    if kind == 'dropout':
        start = int(rng.integers(500,1500)); x[start:start+1000] = 0
    # Arbitrary ADC-like units, not a physiological MAX30102 calibration.
    return rng.uniform(65000,140000) + rng.uniform(800,7000)*x

def features(raw):
    x = np.asarray(raw, dtype=float)
    if x.shape != (3000,) or not np.isfinite(x).all():
        raise ValueError('Expected exactly 3000 finite raw IR samples (30 seconds at 100 Hz).')
    scale = np.std(x)
    if scale < 1e-9: return np.array([0,0,0,0,0,1.])
    z = (x - np.mean(x))/scale
    filtered = sosfiltfilt(butter(3,[.4,4],btype='bandpass',fs=FS,output='sos'),z)
    peaks,_ = find_peaks(filtered,distance=25,prominence=.5)
    intervals = np.diff(peaks)/FS
    bpm = 60/np.median(intervals) if len(intervals) else 0
    cv = np.std(intervals)/np.mean(intervals) if len(intervals) else 0
    spread = np.ptp(intervals) if len(intervals) else 0
    f,p = welch(z,fs=FS,nperseg=1000)
    total = p.sum()+1e-12
    band = p[(f>=.4)&(f<=4)]
    concentration = band.max()/(band.sum()+1e-12)
    return np.array([bpm,cv,spread,concentration,p[f>5].sum()/total,np.mean(np.abs(np.diff(x))<1e-8)])

def build():
    out = ROOT/'data'; out.mkdir(exist_ok=True)
    examples = ROOT/'examples'; examples.mkdir(exist_ok=True)
    records=[]; waves=[]; X=[]; y=[]; split=[]
    # Independently seeded recordings; test recordings never enter training.
    for i,kind in enumerate(KINDS):
        label = 'regular_pattern' if kind=='regular' else ('poor_signal' if kind in ['noise','flat','dropout'] else 'unusual_pattern')
        for j in range(100):
            seed=10000+i*1000+j
            raw=generate(kind,np.random.default_rng(seed)); feat=features(raw)
            partition='train' if j<80 else 'test'
            records.append([f'SYN_{kind}_{j:03d}',True,kind,label,partition,seed,*feat])
            waves.append(raw); X.append(feat); y.append(label); split.append(partition)
            if j==99:
                with (examples/f'SYNTHETIC_{kind}.csv').open('w',newline='') as f:
                    writer=csv.writer(f); writer.writerow(['sample_index','sample_time_us_nominal','read_elapsed_us','ir','finger_present_hint'])
                    writer.writerows((k,k*10000,k*10000,float(v),int(v>50000)) for k,v in enumerate(raw))
    with (out/'synthetic_manifest_features.csv').open('w',newline='') as f:
        writer=csv.writer(f); writer.writerow(['recording_id','is_synthetic','simulated_pattern','target','split','seed',*FEATURES]);writer.writerows(records)
    np.savez_compressed(out/'synthetic_raw_windows.npz',ir=np.array(waves),sampling_hz=FS,recording_id=np.array([r[0] for r in records]),is_synthetic=True)
    X=np.array(X); y=np.array(y); train=np.array(split)=='train'
    model=RandomForestClassifier(n_estimators=150,max_depth=10,class_weight='balanced',random_state=42,n_jobs=-1)
    model.fit(X[train],y[train]); predicted=model.predict(X[~train])
    report={'scope':'HELD-OUT SYNTHETIC RECORDINGS ONLY. No real-world or clinical validation.', 'recordings':700,'train':560,'test':140,'sample_rate_hz':100,'seconds_per_window':30,'classes':model.classes_.tolist(),'classification_report':classification_report(y[~train],predicted,output_dict=True,zero_division=0),'confusion_matrix':confusion_matrix(y[~train],predicted,labels=model.classes_).tolist(),'sklearn_version':sklearn.__version__}
    (ROOT/'synthetic_evaluation.json').write_text(json.dumps(report,indent=2))
    with (ROOT/'synthetic_pattern_model.pkl').open('wb') as f:
        pickle.dump({'model':model,'features':FEATURES,'sample_rate_hz':FS,'window_samples':3000,'synthetic_only':True,'sklearn_version':sklearn.__version__},f)
    print(json.dumps({'synthetic_test_accuracy':report['classification_report']['accuracy'],'scope':report['scope']},indent=2))

def predict(path):
    with open(path,newline='') as f:
        rows=list(csv.DictReader(f))
    if len(rows)!=3000: raise ValueError('This demo accepts exactly one complete 30-second, 100-Hz recording.')
    if [int(r['sample_index']) for r in rows]!=list(range(3000)): raise ValueError('Missing or unordered samples.')
    if [int(r['sample_time_us_nominal']) for r in rows]!=[k*10000 for k in range(3000)]: raise ValueError('Expected nominal 100-Hz timing.')
    with (ROOT/'synthetic_pattern_model.pkl').open('rb') as f: bundle=pickle.load(f)
    result=bundle['model'].predict([features([float(r['ir']) for r in rows])])[0]
    return {'pattern':result,'experimental':True,'trained_on':'synthetic_only','diagnosis':None,'notice':'Software demonstration only; real-person predictions are unvalidated.'}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['build','predict']);parser.add_argument('csv_path',nargs='?');args=parser.parse_args()
    if args.command=='build': build()
    else:
        if not args.csv_path: parser.error('predict requires a CSV path')
        print(json.dumps(predict(args.csv_path),indent=2))
