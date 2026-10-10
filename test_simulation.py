"""Run locally: python test_simulation.py (uses Flask's in-process test client)."""
import os
import json
from pathlib import Path
import numpy as np
os.environ['API_KEY'] = 'local-test-only'
from app import app
from pulse_demo import generate
client = app.test_client()
headers = {'x-api-key': 'local-test-only'}
results = []
for i, kind in enumerate(['regular','slow','fast','irregular','noise','flat','dropout']):
    raw = generate(kind, np.random.default_rng(10000+i*1000+99)).tolist()
    body = {'source':'synthetic','sample_rate_hz':100,'samples':raw}
    response = client.post('/api/v1/simulation/predict',json=body,headers=headers)
    expected = 'regular_pattern' if kind=='regular' else 'poor_signal' if kind in ['noise','flat','dropout'] else 'unusual_pattern'
    assert response.status_code == 200, response.json
    assert response.json['pattern'] == expected, response.json
    assert (response.json['bpm'] is None) == (expected=='poor_signal')
    results.append({'example':kind,'pattern':response.json['pattern'],'estimated_bpm':response.json['bpm']})
for auth in [{}, {'x-api-key':'wrong'}]:
    assert client.post('/api/v1/simulation/predict',json=body,headers=auth).status_code==401
for bad in [{},[],{**body,'source':'live'},{**body,'samples':raw[:-1]},{**body,'sample_rate_hz':64},{**body,'samples':['1']*3000},{**body,'samples':[float('nan')]*3000}]:
    assert client.post('/api/v1/simulation/predict',json=bad,headers=headers).status_code==400
os.environ.pop('API_KEY')
assert client.post('/api/v1/simulation/predict',json=body,headers=headers).status_code==503
assert client.get('/api/v1/health').status_code==200
report={'backend_tests':'PASS: 7 examples, missing/wrong authentication, invalid source/rate/length/type/nonfinite samples, missing server key, health','examples':results,'android_validation':'Source/XML inspection only; Android SDK unavailable; APK build and device UI not tested.','deployment':'Not deployed','scope':'Synthetic only. No clinical or hardware validation.'}
Path(__file__).with_name('SIMULATION_TEST_RESULTS.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
