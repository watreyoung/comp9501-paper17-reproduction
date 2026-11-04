"""Independently check full adapter/head grid, identities, AP and selection."""
import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
from experiments.core_run import source_captions
from experiments.paper_alignment import domain_cache
from experiments.data import make_task
from experiments.run import sha256_file

HEADS = ['SVM', 'sMIL-PI', 'mi-SVM-PI', 'MIL-CPB-PI']


def audit(root, partial=False):
    root = Path(root)
    manifest = json.loads((root / 'run_manifest.json').read_text())
    config = manifest['config']
    core = Path(config['core_results'])
    assert sha256_file(core / 'run_manifest.json') == manifest['core_manifest_sha256']
    cc = json.loads((core / 'run_manifest.json').read_text())['config']
    assert sha256_file(cc['cache']) == manifest['cache_sha256']
    assert sha256_file(config['protocol_file']) == manifest['protocol_sha256']
    assert sha256_file(root / 'protocol_snapshot.md') == manifest['protocol_sha256']
    assert sha256_file(root / 'config_snapshot.json') == manifest['config_sha256']
    assert json.loads((root / 'config_snapshot.json').read_text()) == config
    for file, digest in manifest['source_hashes'].items():
        assert sha256_file(file) == digest, file
        assert sha256_file(root / 'source_snapshot' / file) == digest, file
    rows = [json.loads(s) for s in (root / 'records.jsonl').read_text().splitlines()]
    selected_path = root / 'selected_parameters.json'
    selected = json.loads(selected_path.read_text()) if selected_path.exists() else {}
    with np.load(cc['cache']) as f:
        cache = {k: f[k] for k in f.files}
    captions, _ = source_captions()
    domains, expected = {}, set()
    with threadpool_limits(limits=1):
        for domain in cc['domains']:
            c, dm = domain_cache(cache, domain, captions=captions, n_bags=cc['n_bags'], bag_size=cc['bag_size'])
            domains[domain] = (c, dm)
            path = root / f'{domain}_domain.json'
            if path.exists():
                assert json.loads(path.read_text()) == dm
            else:
                assert partial, f'Missing domain artifact {path}'
            for adapter, head, cls in itertools.product(config['adapters'], HEADS, dm['classes']):
                for scale in cc['scales']:
                    expected.add((domain, adapter, head, cls, cc['validation_seed'], cc['validation_noise'], 'validate', scale))
                for seed, noise in itertools.product(cc['seeds'], cc['noise_levels']):
                    expected.add((domain, adapter, head, cls, seed, noise, 'test', None))
    seen, tasks, transforms, statuses = set(), {}, {}, {}
    max_violation = 0.
    for row in rows:
        key = (row['domain'], row['adapter'], row['head'], row['class_id'], row['seed'], row['noise'], row['split'], row['scale'] if row['split'] == 'validate' else None)
        assert key in expected and key not in seen, key
        seen.add(key)
        identity = {k: row[k] for k in ['domain', 'adapter', 'head', 'method', 'class_id', 'seed', 'noise', 'split', 'scale']}
        assert row['method'] == row['adapter']+'+'+row['head']
        assert row['record_id'] == hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
        tk = (row['domain'], row['class_id'], row['seed'], row['noise'], row['split'])
        assert row['task_id'] == hashlib.sha256(json.dumps(list(tk)).encode()).hexdigest()[:20]
        if tk not in tasks:
            c = domains[row['domain']][0]
            data, bag = make_task(c, row['class_id'], seed=row['seed'], noise=row['noise'], n_bags=cc['n_bags'], bag_size=cc['bag_size'], evaluation_split=row['split'])
            assert json.loads((root / 'bags' / (row['task_id']+'.json')).read_text()) == bag
            tasks[tk] = (data.test_labels, bag['evaluation_video_ids'])
        statuses[row['status']] = statuses.get(row['status'], 0)+1
        if row['split'] == 'test':
            assert row['scale'] == selected[row['domain']][row['adapter']]['heads'][row['head']]['scale']
        if row['status'] == 'failed':
            continue
        pred = root / 'predictions' / (row['record_id']+'.npz')
        assert sha256_file(pred) == row['prediction_sha256']
        with np.load(pred) as p:
            assert np.isfinite(p['scores']).all()
            np.testing.assert_array_equal(p['labels'], tasks[tk][0])
            np.testing.assert_array_equal(p['video_ids'], tasks[tk][1])
            assert abs(average_precision_score(p['labels'] > 0, p['scores'])-row['average_precision']) < 1e-12
        tf = row['transform_file']
        assert tf == f"transforms/{row['task_id']}_{row['adapter']}.npz"
        if tf not in transforms:
            transforms[tf] = sha256_file(root / tf)
            with np.load(root / tf) as f:
                assert all(np.isfinite(f[k]).all() for k in f.files)
                n = 2*cc['n_bags']*cc['bag_size']
                assert f['training_kernel'].shape == (n, n)
                assert f['evaluation_kernel'].shape == (len(tasks[tk][0]), n)
                np.testing.assert_allclose(f['training_kernel'], f['training_kernel'].T, atol=1e-10)
        assert transforms[tf] == row['transform_sha256']
        violation = row['diagnostics']['max_constraint_violation']
        assert violation <= 1e-6
        max_violation = max(max_violation, violation)
        if row['status'] == 'ok':
            assert row['diagnostics']['stop_reason'] not in ['max_iter', 'line_search_failed']
            assert row['transform_diagnostics'].get('stop_reason') not in ['max_iter', 'line_search_failed']
    # Deliberately do not call the runner's selection function.
    for domain, choices in selected.items():
        classes = set(domains[domain][1]['classes'])
        assert set(choices) == set(config['adapters'])
        for adapter in config['adapters']:
            options = []
            for head in HEADS:
                candidates = []
                for scale in cc['scales']:
                    group = [r for r in rows if (r['domain'], r['adapter'], r['head'], r['scale'], r['split']) == (domain, adapter, head, scale, 'validate')]
                    if {r['class_id'] for r in group} == classes and len(group) == len(classes) and all(r['status'] == 'ok' for r in group):
                        candidates.append(dict(scale=scale, validation_map=float(np.mean([r['average_precision'] for r in group]))))
                assert candidates
                best = max(candidates, key=lambda r: r['validation_map'])
                assert choices[adapter]['heads'][head] == best
                if head != 'SVM':
                    options.append(dict(head=head, **best))
            assert choices[adapter]['family'] == max(options, key=lambda r: r['validation_map'])
    bad_test = sum(r['split'] == 'test' and r['status'] != 'ok' for r in rows)
    complete = seen == expected and not bad_test and set(selected) == set(cc['domains'])
    result = dict(status='grid_and_predictions_passed' if complete else 'partial_not_delivery', expected_fits=len(expected), observed_fits=len(rows), missing_fits=len(expected-seen), status_counts=statuses, bag_tasks_checked=len(tasks), transforms_checked=len(transforms), max_constraint_violation=max_violation, failed_or_incomplete_test_fits=bad_test, classifier_replay='separate requirement; not proved by this audit', transform_recomputation='not performed by this audit')
    if not partial:
        assert complete, result
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--allow-partial', action='store_true')
    args = p.parse_args()
    result = audit(args.root, args.allow_partial)
    output = args.root / ('partial_audit.json' if args.allow_partial else 'grid_prediction_audit.json')
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
