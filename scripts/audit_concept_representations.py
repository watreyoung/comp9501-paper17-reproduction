"""Audit formal concept identities and independently replay sampled scores."""
import argparse
import gc
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from experiments.core_run import source_captions
from experiments.paper_alignment import domain_cache
from experiments.concept_data import miml_source_bags, classeme_word_bags
from experiments.run import sha256_file
from mil_pi.kernels import l2_normalize_columns


def audit(root, partial=False):
    root = Path(root)
    manifest = json.loads((root/'run_manifest.json').read_text())
    config = manifest['config']
    core = Path(config['core_results'])
    assert sha256_file(core/'run_manifest.json') == manifest['core_manifest_sha256']
    cc = json.loads((core/'run_manifest.json').read_text())['config']
    assert sha256_file(cc['cache']) == manifest['cache_sha256']
    assert sha256_file(root/'config_snapshot.json') == manifest['config_sha256']
    assert json.loads((root/'config_snapshot.json').read_text()) == config
    assert sha256_file(config['protocol_file']) == manifest['protocol_sha256']
    assert sha256_file(root/'protocol_snapshot.md') == manifest['protocol_sha256']
    for f, digest in manifest['source_hashes'].items():
        assert sha256_file(f) == digest, f
        assert sha256_file(root/'source_snapshot'/f) == digest, f
    captions, archive_hash = source_captions()
    assert archive_hash == manifest['caption_archive_sha256']
    with np.load(cc['cache']) as f:
        cache = {k:f[k] for k in f.files}
    checked, missing = [], []
    with threadpool_limits(limits=1):
        for domain in cc['domains']:
            c, dm = domain_cache(cache, domain, captions=captions, n_bags=cc['n_bags'], bag_size=cc['bag_size'])
            source = np.flatnonzero(c['splits'] == 'train')
            used = np.isin(c['splits'], ['train', 'validate', 'test'])
            for method in config['methods']:
                prefix = root/f'{domain}_{method}'
                complete = Path(str(prefix)+'_complete.json')
                if not complete.exists():
                    missing.append([domain, method])
                    continue
                assert json.loads((root/f'{domain}_domain.json').read_text()) == dm
                report = json.loads(complete.read_text())
                fit = json.loads(Path(str(prefix)+'_fit.json').read_text())
                assert fit['manifest_sha256'] == sha256_file(root/'run_manifest.json')
                for key, value in fit.items():
                    assert report[key] == value
                assert report['source_videos'] == len(source) and report['words'] == 2000
                assert report['represented_videos'] == int(used.sum())
                model_path = Path(str(prefix)+'.joblib')
                score_path = Path(str(prefix)+'_scores.npz')
                assert sha256_file(model_path) == report['model_sha256']
                assert sha256_file(score_path) == report['scores_sha256']
                bundle = joblib.load(model_path)
                np.testing.assert_array_equal(bundle['source_ids'], c['video_ids'][source])
                model = bundle['model']
                rng = np.random.default_rng(42)
                rows = np.concatenate([rng.choice(np.flatnonzero(c['splits'] == split), min(5, int(np.sum(c['splits'] == split))), replace=False) for split in ['train', 'validate', 'test']])
                if method == 'Classeme':
                    assert model.weights.shape == (c['visual'].shape[1], 2000)
                    assert len(model.diagnostics) == 2000
                    for word, diag in enumerate(model.diagnostics):
                        pos, neg = classeme_word_bags(c['text'], c['splits'], word, n_bags=cc['n_bags'], bag_size=cc['bag_size'], seed=config['seed'])
                        assert diag['positive_rows'] == [b.tolist() for b in pos]
                        assert diag['negative_rows'] == [b.tolist() for b in neg]
                        assert diag['stop_reason'] == 'labels_stable'
                        assert diag['max_constraint_violation'] <= 1e-6
                    # Reconstruct linear decisions directly rather than through model API.
                    predicted = l2_normalize_columns(c['visual'][rows].T).T @ model.weights+model.biases
                else:
                    bag_rows, labels = miml_source_bags(c['text'], c['splits'], bag_size=cc['bag_size'], seed=config['seed'])
                    np.testing.assert_array_equal(labels, bundle['word_labels'])
                    assert len(bag_rows) == len(bundle['bag_rows'])
                    for a, b in zip(bag_rows, bundle['bag_rows']):
                        np.testing.assert_array_equal(a, b)
                    scaler = bundle['scaler']
                    # SourceMinMax fits in float64; float32 subtraction rounds
                    # the span differently even with identical raw source data.
                    raw_source = np.asarray(c['visual'][source], dtype=float)
                    np.testing.assert_array_equal(scaler.minimum, raw_source.min(axis=0))
                    np.testing.assert_array_equal(scaler.span, np.ptp(raw_source, axis=0))
                    assert model.stop_reason == 'medoids_stable'
                    assert len(model.classifiers) == 2000
                    for idx, bag in zip(model.medoid_indices, model.medoid_bags):
                        np.testing.assert_array_equal(bag, scaler.transform(c['visual'][bag_rows[idx]]))
                    assert all(clf is None or clf.fit_status_ == 0 for clf in model.classifiers)
                    X = scaler.transform(c['visual'][rows])
                    predicted = model.decision_function([v[None, :] for v in X])
                with np.load(score_path) as f:
                    np.testing.assert_array_equal(f['video_ids'], c['video_ids'])
                    np.testing.assert_array_equal(f['splits'], c['splits'])
                    np.testing.assert_array_equal(f['represented_mask'], used)
                    assert f['scores'].shape == (len(c['video_ids']), 2000)
                    assert np.isfinite(f['scores']).all()
                    assert not np.any(f['scores'][~used])
                    actual = f['scores'][rows]
                    np.testing.assert_allclose(actual, predicted, rtol=1e-9, atol=1e-10)
                    error = float(np.max(np.abs(actual-predicted)))
                checked.append(dict(domain=domain, method=method, sampled_rows=rows.tolist(), max_score_error=error, all_words=2000))
                del bundle, model
                gc.collect()
    result = dict(status='concept_cache_identity_and_sample_replay_passed' if not missing else 'partial_not_delivery', checked=checked, missing=missing, scope='Concept cache identities and 15 sampled videos per completed representation; not a full prediction replay or event evaluation', script_sha256=sha256_file(__file__))
    if not partial:
        assert not missing, result
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--allow-partial', action='store_true')
    args = p.parse_args()
    result = audit(args.root, args.allow_partial)
    (args.root/('partial_audit.json' if args.allow_partial else 'cache_audit.json')).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
