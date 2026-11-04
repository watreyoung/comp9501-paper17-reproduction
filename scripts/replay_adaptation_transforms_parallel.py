"""Recompute every saved transform using frozen mathematics, in parallel."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from experiments import full_adaptation_parallel as worker
from experiments.full_adaptation_wolfe import feature_adaptation
from experiments.run import sha256_file


def recompute(row):
    ns = worker._NAMESPACE
    root, config, cc = ns['root'], ns['config'], ns['cc']
    domain = row['domain']
    if domain not in worker._DOMAINS:
        worker._DOMAINS[domain] = ns['domain_cache'](worker._CACHE, domain, captions=worker._CAPTIONS, n_bags=cc['n_bags'], bag_size=cc['bag_size'])[0]
    data, bag = ns['make_task'](worker._DOMAINS[domain], row['class_id'], seed=row['seed'], noise=row['noise'], n_bags=cc['n_bags'], bag_size=cc['bag_size'], evaluation_split=row['split'])
    assert json.loads((root/'bags'/(row['task_id']+'.json')).read_text()) == bag
    path = root/row['transform_file']
    assert sha256_file(path) == row['transform_sha256']
    p = ns['prepare_problem'](data)
    adapted, diag, arrays = feature_adaptation(row['adapter'], data, p, dimension=config['dimension'], tca_mu=config['tca_mu_internal'])
    arrays.update(training_kernel=adapted['K'], evaluation_kernel=adapted['test_kernel'])
    error = 0.
    with np.load(path) as saved:
        assert set(saved.files) == set(arrays)
        for key, value in arrays.items():
            np.testing.assert_allclose(value, saved[key], rtol=1e-9, atol=1e-10)
            error = max(error, float(np.max(np.abs(value-saved[key]))))
    return row['transform_file'], dict(transform_sha256=row['transform_sha256'], max_array_error=error, stop_reason=diag.get('stop_reason'))


def replay(root, workers=None):
    root = Path(root)
    manifest = json.loads((root/'run_manifest.json').read_text())
    config = manifest['config']
    assert config['protocol_version'] == 'full_adaptation_parallel'
    for file, digest in manifest['source_hashes'].items():
        assert sha256_file(file) == digest, file
    core = Path(config['core_results'])
    assert sha256_file(core/'run_manifest.json') == manifest['core_manifest_sha256']
    cc = json.loads((core/'run_manifest.json').read_text())['config']
    assert sha256_file(cc['cache']) == manifest['cache_sha256']
    rows = [json.loads(v) for v in (root/'records.jsonl').read_text().splitlines()]
    unique = {}
    for row in rows:
        if row['status'] != 'failed':
            unique.setdefault(row['transform_file'], row)
    out = root/'transform_replay'
    out.mkdir(exist_ok=True)
    with (out/'.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = out/'records.json'
        reports = json.loads(path.read_text()) if path.exists() else {}
        for name, row in unique.items():
            assert sha256_file(root/name) == row['transform_sha256']
            if name in reports:
                assert reports[name]['transform_sha256'] == row['transform_sha256']
        targets = [r for name,r in unique.items() if name not in reports]
        count = workers or config['workers']
        with ProcessPoolExecutor(max_workers=count, initializer=worker.initialize, initargs=(str(root), config, cc, {}, set())) as pool:
            for name, result in pool.map(recompute, targets, chunksize=1):
                reports[name] = result
                tmp = out/'records.tmp'
                tmp.write_text(json.dumps(reports, indent=2)+'\n')
                tmp.replace(path)
                print('recomputed', len(reports), '/', len(unique), name, result['max_array_error'], flush=True)
        summary = dict(recomputed_transforms=len(reports), snapshot_transforms=len(unique), workers=count, batch_complete=False, max_array_error=max((r['max_array_error'] for r in reports.values()), default=0.), script_sha256=sha256_file(__file__), scope='All transforms in this snapshot; independent recomputation of frozen formulas; full grid completion is a separate gate')
        (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--workers', type=int)
    args = p.parse_args()
    print(json.dumps(replay(args.root, args.workers), indent=2))
