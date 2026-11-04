"""Frozen four-adapter/four-head evaluation; validation-only family choice."""
import argparse
import hashlib
import json
import shutil
import time
import traceback
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
from .core_run import source_captions
from .paper_alignment import domain_cache
from .data import make_task
from .models import prepare_problem
from .full_adaptation_wolfe import HEADS, feature_adaptation, classify_adaptation, select_validation_family
from .run import sha256_file, write_json, store_record


def selections(rows, domain, classes, config, scales):
    result = {}
    for adapter in config['adapters']:
        subset = [r for r in rows if r['domain'] == domain and r['adapter'] == adapter and r['split'] == 'validate']
        heads = {}
        for head in HEADS:
            options = []
            for scale in scales:
                group = [r for r in subset if r['head'] == head and r['scale'] == scale]
                ids = [r['class_id'] for r in group]
                if len(ids) != len(set(ids)):
                    raise ValueError('Duplicate validation candidate')
                if set(ids) == set(classes) and all(r['status'] == 'ok' for r in group):
                    options.append(dict(scale=scale, validation_map=float(np.mean([r['average_precision'] for r in group]))))
            if not options:
                raise RuntimeError(f'No complete validation candidate: {domain}/{adapter}/{head}')
            heads[head] = max(options, key=lambda v: v['validation_map'])
        result[adapter] = dict(heads=heads, family=select_validation_family(subset, classes, scales))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/full_adaptation.json'))
    parser.add_argument('--output', type=Path, default=Path('results/full_adaptation'))
    parser.add_argument('--phase', choices=['tune', 'evaluate', 'all'], default='all')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config['adapters'] != ['SA', 'TCA', 'GFK', 'DIP']:
        raise ValueError('This version covers exactly the four verified adapters')
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    core = Path(config['core_results'])
    cm = json.loads((core / 'run_manifest.json').read_text())
    cc = cm['config']
    sources = set(cm['source_hashes']) | set(config['additional_sources']) | {'experiments/full_adaptation_wolfe_run.py'}
    for f, digest in cm['source_hashes'].items():
        assert sha256_file(f) == digest, f
    identity = dict(config=config, config_sha256=sha256_file(args.config), core_manifest_sha256=sha256_file(core / 'run_manifest.json'), cache_sha256=sha256_file(cc['cache']), protocol_sha256=sha256_file(config['protocol_file']), source_hashes={f: sha256_file(f) for f in sorted(sources)})
    mp = root / 'run_manifest.json'
    if mp.exists():
        assert json.loads(mp.read_text()) == identity, 'Frozen identity changed; use new output directory'
    else:
        write_json(mp, identity)
        shutil.copy2(args.config, root / 'config_snapshot.json')
        shutil.copy2(config['protocol_file'], root / 'protocol_snapshot.md')
        for f in sources:
            dest = root / 'source_snapshot' / f
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest)
    for name in ['bags', 'predictions', 'transforms']:
        (root / name).mkdir(exist_ok=True)
    with np.load(cc['cache']) as f:
        cache = {k: f[k] for k in f.files}
    captions, _ = source_captions()
    log = root / 'records.jsonl'
    rows = [json.loads(s) for s in log.read_text().splitlines()] if log.exists() else []
    existing = {r['record_id'] for r in rows}
    selected_path = root / 'selected_parameters.json'
    selected = json.loads(selected_path.read_text()) if selected_path.exists() else {}

    def task(domain, c, cls, seed, noise, split):
        tid = hashlib.sha256(json.dumps([domain, cls, seed, noise, split]).encode()).hexdigest()[:20]
        data, bag = make_task(c, cls, seed=seed, noise=noise, n_bags=cc['n_bags'], bag_size=cc['bag_size'], evaluation_split=split)
        bp = root / 'bags' / f'{tid}.json'
        if bp.exists():
            assert json.loads(bp.read_text()) == bag
        else:
            write_json(bp, bag)
        p = prepare_problem(data)
        text = np.concatenate(data.positive_text_bags + data.negative_text_bags, axis=1)
        p['full_tK'] = text.T @ text
        for adapter in config['adapters']:
            candidates = []
            for head in HEADS:
                scales = cc['scales'] if split == 'validate' else [selected[domain][adapter]['heads'][head]['scale']]
                for scale in scales:
                    key = dict(domain=domain, adapter=adapter, head=head, method=adapter+'+'+head, class_id=cls, seed=seed, noise=noise, split=split, scale=scale)
                    rid = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:20]
                    if rid not in existing:
                        candidates.append((key, rid))
            if not candidates:
                continue
            start = time.perf_counter()
            try:
                ap, td, arrays = feature_adaptation(adapter, data, p, dimension=config['dimension'], tca_mu=config['tca_mu_internal'])
                tp = root / 'transforms' / f'{tid}_{adapter}.npz'
                # Store the actual transformed kernels for independent classifier replay.
                arrays.update(training_kernel=ap['K'], evaluation_kernel=ap['test_kernel'])
                if tp.exists():
                    with np.load(tp) as saved:
                        assert set(saved.files) == set(arrays)
                        for k in arrays:
                            np.testing.assert_allclose(saved[k], arrays[k], rtol=1e-10, atol=1e-12)
                else:
                    np.savez_compressed(tp, **arrays)
                transform_error = None
            except Exception as error:
                transform_error = dict(error=str(error), traceback=traceback.format_exc())
            elapsed = time.perf_counter() - start
            for key, rid in candidates:
                row = dict(key, record_id=rid, task_id=tid, transform_seconds=elapsed)
                try:
                    if transform_error:
                        raise RuntimeError('Transform failed: '+transform_error['error'])
                    scores, diag = classify_adaptation(key['head'], ap, scale=key['scale'], seed=seed)
                    if not np.isfinite(scores).all() or diag['max_constraint_violation'] > 1e-6:
                        raise ArithmeticError('Nonfinite scores or classifier constraint violation')
                    incomplete = td.get('stop_reason') in ['max_iter', 'line_search_failed'] or diag['stop_reason'] in ['max_iter', 'line_search_failed']
                    pred = root / 'predictions' / f'{rid}.npz'
                    np.savez_compressed(pred, scores=scores, labels=data.test_labels, video_ids=bag['evaluation_video_ids'])
                    row.update(status='incomplete' if incomplete else 'ok', average_precision=float(average_precision_score(data.test_labels > 0, scores)), diagnostics=diag, transform_diagnostics=td, transform_file=str(tp.relative_to(root)), transform_sha256=sha256_file(tp), prediction_sha256=sha256_file(pred))
                except Exception as error:
                    row.update(status='failed', error=str(error), traceback=traceback.format_exc(), transform_failure=transform_error)
                store_record(log, rows, row)
                existing.add(rid)
                print(len(rows), domain, key['method'], cls, seed, noise, split, row['status'], flush=True)

    with threadpool_limits(limits=1):
        for domain in cc['domains']:
            c, dm = domain_cache(cache, domain, captions=captions, n_bags=cc['n_bags'], bag_size=cc['bag_size'])
            assert dm == json.loads((core / f'{domain}_domain.json').read_text())
            write_json(root / f'{domain}_domain.json', dm)
            if args.phase in ['tune', 'all']:
                for cls in dm['classes']:
                    task(domain, c, cls, cc['validation_seed'], cc['validation_noise'], 'validate')
                selected[domain] = selections(rows, domain, dm['classes'], config, cc['scales'])
                write_json(selected_path, selected)
            if args.phase in ['evaluate', 'all']:
                assert selected[domain] == selections(rows, domain, dm['classes'], config, cc['scales'])
                for noise in cc['noise_levels']:
                    for seed in cc['seeds']:
                        for cls in dm['classes']:
                            task(domain, c, cls, seed, noise, 'test')
    print('Batch complete; independent audit still required', len(rows), flush=True)


if __name__ == '__main__':
    main()
