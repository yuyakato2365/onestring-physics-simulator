"""Collect the key numbers of a run and compare them with reference/expected.json (if present).
usage: check_results.py OUT_DIR [--write-reference]"""
import os, sys, json, pickle, numpy as np
S = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, S + '/neck'); sys.path.insert(0, S + '/os_simple')
from analyze import metrics
out = sys.argv[1]; ref_path = os.path.join(os.path.dirname(S), 'reference', 'expected.json')
res = {}
pi = os.path.join(out, '1_inflate', 'inflate_summary.json')
if os.path.exists(pi):
    inf = json.load(open(pi)); res['inflate_delta0'] = inf['delta0']; res['inflate_min_sphere_radius_end'] = inf['min_sphere_radius_end']
r = pickle.load(open(os.path.join(out, '2_optcuts', 'result.pkl'), 'rb')); U, F = np.asarray(r['uv']), np.asarray(r['f2'])
X = np.zeros((len(U), 3)); X[F.ravel()] = np.asarray(r['xyz'])[np.asarray(r['f3']).ravel()]
mt = metrics(U, X, F)
res['optcuts_SD'] = float(mt['SD']); res['optcuts_seam_over_R'] = float(mt['seam_rel']); res['optcuts_outline_loops'] = len(mt['loops'])
res['optcuts_components'] = int(mt['components']); res['optcuts_roundness'] = float(mt['roundness'])
om = json.load(open(os.path.join(out, '4_final', 'omega2_summary.json')))
res['omega2_flips'] = om['flips']; res['omega2_size_ratio_target'] = om['fit']['size_ratio_target']
g = json.load(open(os.path.join(out, '4_final', 'grid', 'summary.json')))
res['grid_panels'] = g['panels']; res['grid_bank_mismatch'] = g['bank_mismatch_after_equalising']; res['grid_seam_gap_max'] = g['seam_partner_gap_max']
res['grid_outline_loops_after_weld'] = g['outline_loops_after_weld']
rm = json.load(open(os.path.join(out, '4_final', 'remesh3d_summary.json')))
res['m3d_panels'] = int(len(pickle.load(open(os.path.join(out, '4_final', 'remesh3d.pkl'), 'rb'))['T']))
for k in ('angle_dev_mean', 'angle_dev_p95', 'angle_min', 'angle_max', 'size_ratio_p95_p5'): res['m3d_' + k] = rm['after'][k]
res['m3d_crossing_panels'] = rm['crossing_panels_end']; res['m3d_fins_removed'] = rm['fins_removed']
print(json.dumps(res, indent=1))
if '--write-reference' in sys.argv:
    json.dump(res, open(ref_path, 'w'), indent=1); print('reference written')
elif os.path.exists(ref_path):
    ref = json.load(open(ref_path)); print('\ncomparison with reference/expected.json (relative difference):')
    for k, v in ref.items():
        a = res.get(k); d = abs(a - v) / max(abs(v), 1e-9) if isinstance(v, (int, float)) and a is not None else None
        print(f'  {k:32s} reference {v!s:>12.12s}  this run {a!s:>12.12s}  ' + ('' if d is None else ('ok' if d < 0.05 else f'DIFFERS ({d:.1%})')))
