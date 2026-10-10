"""Raw-waveform simulation endpoint; never writes patient history."""
import os
import hmac
import pickle
from pathlib import Path
import numpy as np
from flask import Blueprint, request, jsonify
from pulse_demo import features

simulation = Blueprint('simulation', __name__)
with (Path(__file__).parent / 'synthetic_pattern_model.pkl').open('rb') as handle:
    bundle = pickle.load(handle)

@simulation.post('/api/v1/simulation/predict')
def simulation_predict():
    key = os.environ.get('API_KEY', '')
    if not key:
        return jsonify(success=False, message='Set API_KEY on the server.'), 503
    if not hmac.compare_digest(request.headers.get('x-api-key', ''), key):
        return jsonify(success=False, message='Unauthorized'), 401
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(success=False, message='Expected a JSON object.'), 400
    if data.get('source') != 'synthetic' or data.get('sample_rate_hz') != 100:
        return jsonify(success=False, message='This endpoint requires synthetic data at 100 Hz.'), 400
    raw = data.get('samples')
    if not isinstance(raw, list) or len(raw) != 3000 or any(type(v) not in (int, float) for v in raw):
        return jsonify(success=False, message='Expected 3000 numeric raw IR samples.'), 400
    if not np.isfinite(raw).all():
        return jsonify(success=False, message='Samples must be finite.'), 400
    vector = features(raw)
    pattern = str(bundle['model'].predict([vector])[0])
    labels = {'regular_pattern': 'Regular pattern', 'unusual_pattern': 'Unusual pattern', 'poor_signal': 'Poor signal'}
    bpm = round(float(vector[0]), 1) if pattern != 'poor_signal' and vector[0] > 0 else None
    return jsonify(success=True, source='synthetic', experimental=True,
                   pattern=pattern, prediction=labels[pattern], bpm=bpm,
                   diagnosis=None, confidence=None,
                   notice='Simulation — synthetic data. Not a clinical diagnosis.')
