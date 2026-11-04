"""Frozen source-only full-vocabulary concept model and score caches."""
import argparse
import gc
import importlib.metadata
import json
import shutil
import time
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from .core_run import source_captions
from .paper_alignment import domain_cache
from .concept_data import SourceMinMax, miml_source_bags
from .classeme_baseline import fit_classeme
from .mimlsvm_baseline import fit_mimlsvm
from .mimlsvm_shared_inference import compile_shared_mimlsvm
from .run import sha256_file, write_json
from mil_pi.kernels import l2_normalize_columns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/concept_representations.json'))
    parser.add_argument('--output', type=Path, default=Path('results/concept_representations'))
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    assert config['methods'] == ['Classeme', 'MIMLSVM']
    core = Path(config['core_results'])
    cm = json.loads((core/'run_manifest.json').read_text())
    cc = cm['config']
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    files = set(cm['source_hashes']) | {
        'experiments/concept_representation_shared_run.py', 'experiments/concept_representation_run.py', 'experiments/mimlsvm_shared_inference.py', 'experiments/concept_data.py',
        'experiments/classeme_baseline.py', 'experiments/mimlsvm_baseline.py',
        'experiments/hausdorff_blocks.py'}
    for file, digest in cm['source_hashes'].items():
        assert sha256_file(file) == digest, file
    captions, archive_hash = source_captions()
    identity = dict(config=config, config_sha256=sha256_file(args.config),
                    core_manifest_sha256=sha256_file(core/'run_manifest.json'),
                    cache_sha256=sha256_file(cc['cache']), caption_archive_sha256=archive_hash,
                    protocol_sha256=sha256_file(config['protocol_file']),
                    source_hashes={f:sha256_file(f) for f in sorted(files)},
                    versions={n:importlib.metadata.version(n) for n in ['numpy', 'scipy', 'scikit-learn', 'joblib', 'cvxpy', 'osqp']})
    path = root/'run_manifest.json'
    if path.exists():
        assert json.loads(path.read_text()) == identity, 'Frozen cache identity changed'
    else:
        write_json(path, identity)
        shutil.copy2(args.config, root/'config_snapshot.json')
        shutil.copy2(config['protocol_file'], root/'protocol_snapshot.md')
        for file in files:
            dest = root/'source_snapshot'/file
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, dest)
    with np.load(cc['cache']) as f:
        cache = {k:f[k] for k in f.files}
    with threadpool_limits(limits=1):
        for domain in cc['domains']:
            c, dm = domain_cache(cache, domain, captions=captions, n_bags=cc['n_bags'], bag_size=cc['bag_size'])
            assert dm == json.loads((core/f'{domain}_domain.json').read_text())
            write_json(root/f'{domain}_domain.json', dm)
            source = np.flatnonzero(c['splits'] == 'train')
            used = np.flatnonzero(np.isin(c['splits'], ['train', 'validate', 'test']))
            assert c['text'].shape[1] == 2000
            for method in config['methods']:
                prefix = root/f'{domain}_{method}'
                checkpoint = Path(str(prefix)+'_fit.json')
                complete = Path(str(prefix)+'_complete.json')
                model_path = Path(str(prefix)+'.joblib')
                score_path = Path(str(prefix)+'_scores.npz')
                if complete.exists():
                    report = json.loads(complete.read_text())
                    assert sha256_file(model_path) == report['model_sha256']
                    assert sha256_file(score_path) == report['scores_sha256']
                    print('Already complete', domain, method, flush=True)
                    continue
                reuse = Path(config['reuse_fit_root'])
                old_checkpoint = reuse/f'{domain}_{method}_fit.json'
                if not checkpoint.exists() and old_checkpoint.exists():
                    original = json.loads((reuse/'run_manifest.json').read_text())
                    assert original['core_manifest_sha256'] == identity['core_manifest_sha256']
                    assert original['cache_sha256'] == identity['cache_sha256']
                    assert original['caption_archive_sha256'] == identity['caption_archive_sha256']
                    for key in ['methods','seed','classeme_C','mimlsvm_C','mimlsvm_gamma','medoid_fraction','concept_weight']:
                        assert original['config'][key] == config[key], key
                    for file, digest in original['source_hashes'].items():
                        assert sha256_file(file) == digest, file
                    old_fit = json.loads(old_checkpoint.read_text())
                    assert old_fit['manifest_sha256'] == sha256_file(reuse/'run_manifest.json')
                    assert (old_fit['domain'], old_fit['method'], old_fit['words']) == (domain, method, 2000)
                    old_model = reuse/f'{domain}_{method}.joblib'
                    assert sha256_file(old_model) == old_fit['model_sha256']
                    temp = Path(str(model_path)+'.tmp')
                    shutil.copy2(old_model, temp)
                    temp.replace(model_path)
                    assert sha256_file(model_path) == old_fit['model_sha256']
                    write_json(checkpoint, dict(old_fit, manifest_sha256=sha256_file(path), reused_fit_from=str(reuse), original_fit_report_sha256=sha256_file(old_checkpoint), original_manifest_sha256=sha256_file(reuse/'run_manifest.json')))
                    print('Reused verified fitted model', domain, method, flush=True)
                if checkpoint.exists():
                    fit_report = json.loads(checkpoint.read_text())
                    assert sha256_file(model_path) == fit_report['model_sha256']
                    bundle = joblib.load(model_path)
                    np.testing.assert_array_equal(bundle['source_ids'], c['video_ids'][source])
                    print('Resume inference', domain, method, flush=True)
                else:
                    print('Fit', domain, method, len(source), 'source videos, 2000 words', flush=True)
                    start = time.perf_counter()
                    if method == 'Classeme':
                        visual = l2_normalize_columns(c['visual'].T).T
                        model = fit_classeme(visual, c['text'], c['splits'], C=config['classeme_C'], seed=config['seed'], n_bags=cc['n_bags'], bag_size=cc['bag_size'])
                        assert all(d['stop_reason'] == 'labels_stable' and d['max_constraint_violation'] <= 1e-6 for d in model.diagnostics)
                        bundle = dict(model=model, scaler=None, source_ids=c['video_ids'][source])
                    else:
                        scaler = SourceMinMax.fit(c['visual'][source])
                        bag_rows, labels = miml_source_bags(c['text'], c['splits'], bag_size=cc['bag_size'], seed=config['seed'])
                        bags = [scaler.transform(c['visual'][b]) for b in bag_rows]
                        model = fit_mimlsvm(bags, labels, medoid_fraction=config['medoid_fraction'], C=config['mimlsvm_C'], gamma=config['mimlsvm_gamma'], seed=config['seed'])
                        assert model.stop_reason == 'medoids_stable'
                        bundle = dict(model=model, scaler=scaler, source_ids=c['video_ids'][source], bag_rows=bag_rows, word_labels=labels)
                        del bags
                    elapsed = time.perf_counter()-start
                    temp = Path(str(model_path)+'.tmp')
                    joblib.dump(bundle, temp, compress=3)
                    temp.replace(model_path)
                    fit_report = dict(domain=domain, method=method, source_videos=len(source), words=2000, fit_seconds=elapsed, model_sha256=sha256_file(model_path), manifest_sha256=sha256_file(path))
                    write_json(checkpoint, fit_report)
                    del model
                compiled = compile_shared_mimlsvm(bundle['model']) if method == 'MIMLSVM' else None
                scores = np.zeros((len(c['video_ids']), 2000))
                start = time.perf_counter()
                # Small deterministic chunks bound representation inference memory.
                for offset in range(0, len(used), 256):
                    rows = used[offset:offset+256]
                    if method == 'Classeme':
                        X = l2_normalize_columns(c['visual'][rows].T).T
                        scores[rows] = bundle['model'].decision_function(X)
                    else:
                        X = bundle['scaler'].transform(c['visual'][rows])
                        scores[rows] = compiled.decision_function([v[None, :] for v in X])
                    assert np.isfinite(scores[rows]).all()
                np.savez_compressed(score_path, scores=scores, video_ids=c['video_ids'], splits=c['splits'], represented_mask=np.isin(np.arange(len(scores)), used))
                write_json(complete, dict(fit_report, scores_sha256=sha256_file(score_path), represented_videos=len(used), prediction_seconds=time.perf_counter()-start, event_AP_not_evaluated=True, audit_status='independent audit still required'))
                print('Complete', domain, method, flush=True)
                del bundle, scores, compiled
                gc.collect()


if __name__ == '__main__':
    main()
