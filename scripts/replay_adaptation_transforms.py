"""Recompute saved feature transforms from raw source/target visual data."""
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


def replay(root, one_per_adapter=False):
    root = Path(root)
    manifest = json.loads((root/'run_manifest.json').read_text())
    config = manifest['config']
    for file, digest in manifest['source_hashes'].items():
        assert sha256_file(file) == digest, file
    if config['protocol_version'] == 'full_adaptation':
        from experiments.full_adaptation_wolfe import feature_adaptation
    elif config['protocol_version'] == 'full_adaptation':
        from experiments.full_adaptation import feature_adaptation
    else:
        raise ValueError('Unsupported frozen transform version')
    core = Path(config['core_results'])
    assert sha256_file(core/'run_manifest.json') == manifest['core_manifest_sha256']
    cc = json.loads((core/'run_manifest.json').read_text())['config']
    assert sha256_file(cc['cache']) == manifest['cache_sha256']
    with np.load(cc['cache']) as f:
        cache = {k:f[k] for k in f.files}
    captions, _ = source_captions()
    rows = [json.loads(s) for s in (root/'records.jsonl').read_text().splitlines()]
    unique = {}
    for row in rows:
        if row['status'] != 'failed':
            unique.setdefault(row['transform_file'], row)
    targets, seen = [], set()
    for name, row in unique.items():
        key = (row['domain'], row['adapter'])
        if not one_per_adapter or key not in seen:
            targets.append((name, row))
            seen.add(key)
    out = root/'transform_replay'
    out.mkdir(exist_ok=True)
    path = out/'records.json'
    reports = json.loads(path.read_text()) if path.exists() else {}
    domains = {}
    with threadpool_limits(limits=1):
        for name, row in targets:
            digest = sha256_file(root/name)
            assert digest == row['transform_sha256']
            if name in reports:
                assert reports[name]['transform_sha256'] == digest
                continue
            domain = row['domain']
            if domain not in domains:
                domains[domain] = domain_cache(cache, domain, captions=captions, n_bags=cc['n_bags'], bag_size=cc['bag_size'])[0]
            data, bag = make_task(domains[domain], row['class_id'], seed=row['seed'], noise=row['noise'], n_bags=cc['n_bags'], bag_size=cc['bag_size'], evaluation_split=row['split'])
            assert json.loads((root/'bags'/(row['task_id']+'.json')).read_text()) == bag
            p = prepare_problem(data)
            transformed, diag, arrays = feature_adaptation(row['adapter'], data, p, dimension=config['dimension'], tca_mu=config['tca_mu_internal'])
            arrays.update(training_kernel=transformed['K'], evaluation_kernel=transformed['test_kernel'])
            error = 0.
            with np.load(root/name) as f:
                assert set(f.files) == set(arrays)
                for key, value in arrays.items():
                    np.testing.assert_allclose(value, f[key], rtol=1e-9, atol=1e-10)
                    error = max(error, float(np.max(np.abs(value-f[key]))))
            reports[name] = dict(transform_sha256=digest, max_array_error=error, stop_reason=diag.get('stop_reason'))
            path.write_text(json.dumps(reports, indent=2)+'\n')
    summary = dict(recomputed_transforms=len(reports), snapshot_transforms=len(unique), one_per_adapter=one_per_adapter, batch_complete=False, max_array_error=max((r['max_array_error'] for r in reports.values()), default=0.), script_sha256=sha256_file(__file__))
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--one-per-adapter', action='store_true')
    args = p.parse_args()
    print(json.dumps(replay(args.root, args.one_per_adapter), indent=2))
