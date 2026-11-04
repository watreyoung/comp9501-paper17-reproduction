"""Replay successful classifiers from saved adapted kernels and source bags."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from threadpoolctl import threadpool_limits
from experiments.core_run import source_captions
from experiments.paper_alignment import domain_cache
from experiments.data import make_task
from experiments.models import prepare_problem
from experiments.full_adaptation import classify_adaptation
from experiments.run import sha256_file


def replay(root, one_per_combination=False):
    root = Path(root)
    manifest = json.loads((root / 'run_manifest.json').read_text())
    for file, digest in manifest['source_hashes'].items():
        assert sha256_file(file) == digest, file
    core = Path(manifest['config']['core_results'])
    cc = json.loads((core / 'run_manifest.json').read_text())['config']
    assert sha256_file(cc['cache']) == manifest['cache_sha256']
    with np.load(cc['cache']) as f:
        cache = {k: f[k] for k in f.files}
    captions, _ = source_captions()
    rows = [json.loads(s) for s in (root / 'records.jsonl').read_text().splitlines()]
    eligible = [r for r in rows if r['status'] == 'ok']
    selected, seen = [], set()
    for r in eligible:
        key = (r['domain'], r['adapter'], r['head'])
        if not one_per_combination or key not in seen:
            selected.append(r)
            seen.add(key)
    out = root / 'classifier_replay'
    out.mkdir(exist_ok=True)
    path = out / 'records.json'
    reports = json.loads(path.read_text()) if path.exists() else {}
    domains, tasks = {}, {}
    with threadpool_limits(limits=1):
        for r in selected:
            rid = r['record_id']
            if rid in reports:
                assert reports[rid]['prediction_sha256'] == r['prediction_sha256']
                assert reports[rid]['transform_sha256'] == r['transform_sha256']
                continue
            domain = r['domain']
            if domain not in domains:
                domains[domain] = domain_cache(cache, domain, captions=captions, n_bags=cc['n_bags'], bag_size=cc['bag_size'])[0]
            tid = r['task_id']
            if tid not in tasks:
                data, bag = make_task(domains[domain], r['class_id'], seed=r['seed'], noise=r['noise'], n_bags=cc['n_bags'], bag_size=cc['bag_size'], evaluation_split=r['split'])
                assert bag == json.loads((root / 'bags' / (tid+'.json')).read_text())
                p = prepare_problem(data)
                text = np.concatenate(data.positive_text_bags+data.negative_text_bags, axis=1)
                p['full_tK'] = text.T @ text
                if len(tasks) >= 2:
                    tasks.clear()
                tasks[tid] = p
            p = dict(tasks[tid])
            assert sha256_file(root / r['transform_file']) == r['transform_sha256']
            with np.load(root / r['transform_file']) as f:
                p.update(K=f['training_kernel'], test_kernel=f['evaluation_kernel'], Kst=f['evaluation_kernel'].T)
            scores, diag = classify_adaptation(r['head'], p, scale=r['scale'], seed=r['seed'])
            assert diag['max_constraint_violation'] <= 1e-6
            pred_path = root / 'predictions' / (rid+'.npz')
            assert sha256_file(pred_path) == r['prediction_sha256']
            with np.load(pred_path) as f:
                error = float(np.max(np.abs(scores-f['scores'])))
                np.testing.assert_allclose(scores, f['scores'], rtol=1e-7, atol=1e-7)
            reports[rid] = dict(prediction_sha256=r['prediction_sha256'], transform_sha256=r['transform_sha256'], max_prediction_error=error)
            path.write_text(json.dumps(reports, indent=2)+'\n')
    result = dict(replayed_fits=len(reports), snapshot_eligible_fits=len(eligible), one_per_combination=one_per_combination, batch_complete=False, max_prediction_error=max((v['max_prediction_error'] for v in reports.values()), default=0.), scope='Classifier replay from saved kernels; transformation recomputation remains separate', script_sha256=sha256_file(__file__))
    (out / 'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--one-per-combination', action='store_true')
    args = p.parse_args()
    print(json.dumps(replay(args.root, args.one_per_combination), indent=2))
