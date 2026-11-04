"""Report the complete frozen adapter grid only after all replay gates pass."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_full_adaptation import audit
from experiments.run import sha256_file


def require_replay_coverage(rows, classifiers, transforms):
    """Check record identities, not only potentially stale replay counters."""
    eligible = [r for r in rows if r['status'] != 'failed']
    assert all(r['status'] == 'ok' for r in eligible), 'Incomplete fit remains'
    for row in eligible:
        c = classifiers[row['record_id']]
        assert c['prediction_sha256'] == row['prediction_sha256']
        assert c['transform_sha256'] == row['transform_sha256']
        assert 0 <= c['max_prediction_error'] <= 1e-7
        t = transforms[row['transform_file']]
        assert t['transform_sha256'] == row['transform_sha256']
        assert 0 <= t['max_array_error'] <= 1e-8
        assert t['stop_reason'] not in ['line_search_failed', 'max_iter']
    return dict(classifiers=len(eligible), transforms=len({r['transform_file'] for r in eligible}))


def report(root):
    root = Path(root)
    # Incomplete batches fail here, before any report file is created.
    checked = audit(root)
    read = lambda name: json.loads((root / name).read_text())
    rows = [json.loads(s) for s in (root / 'records.jsonl').read_text().splitlines()]
    replay = require_replay_coverage(rows, read('classifier_replay/records.json'), read('transform_replay/records.json'))
    manifest = read('run_manifest.json')
    core = Path(manifest['config']['core_results'])
    cc = json.loads((core / 'run_manifest.json').read_text())['config']
    controls = [json.loads(s) for s in (core / 'records.jsonl').read_text().splitlines()]
    heads = ['SVM', 'sMIL-PI', 'mi-SVM-PI', 'MIL-CPB-PI']
    summary, paired = [], []
    for domain in cc['domains']:
        classes = set(read(f'{domain}_domain.json')['classes'])
        for noise in cc['noise_levels']:
            for adapter in manifest['config']['adapters']:
                for head in heads:
                    maps, base_maps = [], []
                    for seed in cc['seeds']:
                        common = lambda r: (r['domain'], r['noise'], r['seed'], r['split']) == (domain, noise, seed, 'test')
                        group = [r for r in rows if common(r) and r['adapter'] == adapter and r['head'] == head]
                        baseline = [r for r in controls if common(r) and r['method'] == head]
                        for part in [group, baseline]:
                            assert len(part) == len(classes) and {r['class_id'] for r in part} == classes
                            assert all(r['status'] == 'ok' for r in part)
                        # A same-class/seed comparison also requires the same actual bags and targets.
                        by_class = {r['class_id']: r for r in baseline}
                        for row in group:
                            b = by_class[row['class_id']]
                            assert all(row[k] == b[k] for k in ['domain', 'class_id', 'seed', 'noise', 'split'])
                            core_bag = json.loads((core / 'bags' / (row['task_id'] + '.json')).read_text())
                            assert core_bag.pop('domain') == domain
                            assert core_bag.pop('task_id') == row['task_id']
                            assert json.loads((root / 'bags' / (row['task_id'] + '.json')).read_text()) == core_bag
                        maps.append(float(np.mean([r['average_precision'] for r in group])))
                        base_maps.append(float(np.mean([r['average_precision'] for r in baseline])))
                    delta = 100 * (np.array(maps) - base_maps)
                    summary.append(dict(domain=domain, noise=noise, method=adapter+'+'+head, seeds=cc['seeds'], seed_mAPs=maps, mean_mAP=float(np.mean(maps)), std_mAP=float(np.std(maps, ddof=1))))
                    paired.append(dict(domain=domain, noise=noise, method=adapter+'+'+head, control=head, seeds=cc['seeds'], seed_delta_pp=delta.tolist(), mean_delta_pp=float(delta.mean()), positive_seeds=int((delta > 1e-6).sum())))
    lines = ['# Full adaptation comparison', '', 'This is a controlled public-data comparison and does not claim numerical reproduction of the original paper or a state-of-the-art result.', '',
             f"The complete grid contains {checked['observed_fits']} fits, with {replay['classifiers']} classifier replays and {replay['transforms']} transform replays. Failed validation candidates are retained and excluded from selection.", '',
             'Values are five-seed mAP means ± sample standard deviation in percentage points. Paired per-seed differences use the same classifier and event bags. Label sets differ between domains and are not compared directly.', '',
             'Adaptation methods use unlabeled target features. Validation and test transforms are fitted separately without target labels. Dimensions, bandwidths, and solver settings follow the frozen protocol.', '']
    for domain in cc['domains']:
        lines += [f'## {domain}', '', '| Method | 0% contamination | 20% contamination | 40% contamination |', '| --- | --- | --- | --- |']
        for adapter in manifest['config']['adapters']:
            for head in heads:
                values = [next(r for r in summary if (r['domain'], r['noise'], r['method']) == (domain, n, adapter+'+'+head)) for n in cc['noise_levels']]
                lines.append('| '+adapter+'+'+head+' | '+' | '.join(f"{100*r['mean_mAP']:.2f} ± {100*r['std_mAP']:.2f}" for r in values)+' |')
        lines.append('')
    lines += ['This adaptation grid does not represent completion of every original-paper baseline. SGF, the specified MILBoost variant, and full MIMLBoost remain outside the saved final evidence. Positive-seed counts are descriptive and are not significance tests.']
    evidence = dict(grid_audit=checked, replay_coverage=replay, script_sha256=sha256_file(__file__), records_sha256=sha256_file(root/'records.jsonl'), classifier_replay_sha256=sha256_file(root/'classifier_replay/records.json'), transform_replay_sha256=sha256_file(root/'transform_replay/records.json'))
    for name, obj in [('summary.json', summary), ('paired_comparisons.json', paired), ('delivery_audit.json', evidence)]:
        (root/name).write_text(json.dumps(obj, indent=2)+'\n')
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return root/'REPORT.md'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    print(report(parser.parse_args().root))
