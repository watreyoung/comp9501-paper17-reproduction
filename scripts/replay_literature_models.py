"""Reconstruct predictions from serialized coefficients, without refitting."""
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
from experiments.run import sha256_file
from mil_pi.kernels import l2_normalize_columns, get_kernel, KernelSpec


def replay(root):
    root = Path(root)
    manifest = json.loads((root/'run_manifest.json').read_text())
    for file, digest in manifest['source_hashes'].items():
        assert sha256_file(file) == digest, file
    core = Path(manifest['config']['core_results'])
    assert sha256_file(core/'run_manifest.json') == manifest['core_manifest_sha256']
    cc = json.loads((core/'run_manifest.json').read_text())['config']
    assert sha256_file(cc['cache']) == manifest['cache_sha256']
    with np.load(cc['cache']) as f:
        cache = {k:f[k] for k in f.files}
    captions, _ = source_captions()
    rows = [json.loads(s) for s in (root/'records.jsonl').read_text().splitlines()]
    out = root/'serialized_model_replay'
    out.mkdir(exist_ok=True)
    path = out/'records.json'
    reports = json.loads(path.read_text()) if path.exists() else {}
    dasvm_dir = root/'reconstructed_dasvm_models'
    dasvm = json.loads((dasvm_dir/'replay_records.json').read_text())
    domains, tasks = {}, {}
    with threadpool_limits(limits=1):
        for row in rows:
            if row['status'] == 'failed':
                continue
            rid = row['record_id']
            model_path = root/'models'/(rid+'.npz')
            pred_path = root/'predictions'/(rid+'.npz')
            assert sha256_file(model_path) == row['model_sha256']
            assert sha256_file(pred_path) == row['prediction_sha256']
            if rid in reports:
                assert reports[rid]['model_sha256'] == row['model_sha256']
                assert reports[rid]['prediction_sha256'] == row['prediction_sha256']
                continue
            domain = row['domain']
            if domain not in domains:
                domains[domain] = domain_cache(cache, domain, captions=captions, n_bags=cc['n_bags'], bag_size=cc['bag_size'])[0]
            tid = row['task_id']
            if tid not in tasks:
                data, bag = make_task(domains[domain], row['class_id'], seed=row['seed'], noise=row['noise'], n_bags=cc['n_bags'], bag_size=cc['bag_size'], evaluation_split=row['split'])
                assert bag == json.loads((root/'bags'/(tid+'.json')).read_text())
                p = prepare_problem(data)
                X = l2_normalize_columns(np.concatenate(data.positive_visual_bags+data.negative_visual_bags, axis=1)).T
                target = l2_normalize_columns(data.test_features).T
                tasks.clear()
                tasks[tid] = p, X, target
            p, X, target = tasks[tid]
            method = row['method']
            with np.load(model_path) as m:
                if method == 'KMM+SVM':
                    scores = p['test_kernel']@m['coef']+m['bias']
                elif method == 'STM':
                    scores = p['test_kernel']@m['coef']
                elif method == 'RT':
                    scores = target@m['visual_weights']
                elif method == 'SVM-uniform':
                    scores = p['test_kernel'][:, m['support_indices']]@m['dual_coef'].ravel()+m['intercept'].item()
                elif method == 'DASVM':
                    supplement = dasvm_dir/(rid+'.npz')
                    assert sha256_file(supplement) == dasvm[rid]['reconstructed_sha256']
                    with np.load(supplement) as saved:
                        coef, bias = saved['joint_coef'], saved['bias'].item()
                        support = np.flatnonzero(coef)
                        joint = np.r_[X, target]
                        scores = np.full(len(target), bias)
                        if len(support):
                            cross = get_kernel(target.T, joint[support].T, KernelSpec('gaussian', saved['kernel_gamma'].item()))
                            scores += cross@coef[support]
                else:
                    raise ValueError('Unknown serialized method '+method)
            with np.load(pred_path) as saved:
                np.testing.assert_allclose(scores, saved['scores'], rtol=1e-7, atol=1e-7)
                error = float(np.max(np.abs(scores-saved['scores'])))
            reports[rid] = dict(method=method, model_sha256=row['model_sha256'], prediction_sha256=row['prediction_sha256'], max_prediction_error=error)
            path.write_text(json.dumps(reports, indent=2)+'\n')
    summary = dict(replayed_fits=len(reports), snapshot_successful_fits=sum(r['status'] != 'failed' for r in rows), max_prediction_error=max((r['max_prediction_error'] for r in reports.values()), default=0.), scope='Serialized coefficients forward replay; grid completeness requires separate audit', script_sha256=sha256_file(__file__))
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    print(json.dumps(replay(p.parse_args().root), indent=2))
