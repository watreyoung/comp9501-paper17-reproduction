"""Frozen mathematics with ordered parallel task execution and provenance."""
import argparse
import itertools
import json
from pathlib import Path
import shutil

from .full_adaptation_parallel import run_tasks
from .full_adaptation_wolfe_run import selections
from .run import sha256_file, write_json, store_record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=Path('configs/full_adaptation_parallel.json'))
    p.add_argument('--output', type=Path, default=Path('results/full_adaptation_parallel'))
    p.add_argument('--import-snapshot', type=Path)
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    core = Path(config['core_results'])
    cm = json.loads((core/'run_manifest.json').read_text())
    cc = cm['config']
    sources = set(cm['source_hashes']) | set(config['additional_sources']) | {'experiments/full_adaptation_wolfe_run.py', 'experiments/full_adaptation_parallel.py', 'experiments/full_adaptation_parallel_run.py'}
    for file, digest in cm['source_hashes'].items():
        assert sha256_file(file) == digest, file
    identity = dict(config=config, config_sha256=sha256_file(args.config), core_manifest_sha256=sha256_file(core/'run_manifest.json'), cache_sha256=sha256_file(cc['cache']), protocol_sha256=sha256_file(config['protocol_file']), source_hashes={f:sha256_file(f) for f in sorted(sources)})
    root = args.output
    root.mkdir(exist_ok=True, parents=True)
    mp = root/'run_manifest.json'
    if mp.exists():
        assert json.loads(mp.read_text()) == identity, 'Frozen identity changed'
        assert not args.import_snapshot, 'Import is only allowed on initialization'
    else:
        assert not (root/'records.jsonl').exists()
        if args.import_snapshot:
            source = args.import_snapshot
            old = json.loads((source/'run_manifest.json').read_text())
            assert old['config']['protocol_version'] == 'full_adaptation'
            assert old['core_manifest_sha256'] == identity['core_manifest_sha256']
            assert old['cache_sha256'] == identity['cache_sha256']
            for key in ['adapters', 'dimension', 'tca_mu_internal', 'core_results']:
                assert old['config'][key] == config[key]
            for file, digest in old['source_hashes'].items():
                assert identity['source_hashes'][file] == digest, file
            parity = json.loads(Path(config['parity_evidence']).read_text())
            assert parity['status'] == 'parallel_frozen_task_parity_passed'
            assert parity['source_manifest_sha256'] == sha256_file(source/'run_manifest.json')
            assert parity['executor_sha256'] == sha256_file('experiments/full_adaptation_parallel.py')
            # Source runner must be stopped before copying. Reject a live writer.
            for proc in Path('/proc').iterdir():
                if not proc.name.isdigit():
                    continue
                try:
                    parts = (proc/'cmdline').read_bytes().split(b'\0')
                    if b'experiments.full_adaptation_wolfe_run' in parts and b'-m' in parts and (proc/'cwd').resolve() == Path.cwd().resolve():
                        raise RuntimeError('Stop the snapshot writer before importing its snapshot')
                except (FileNotFoundError, ProcessLookupError, PermissionError):
                    pass
            source_digest = sha256_file(source/'records.jsonl')
            old_rows = [json.loads(v) for v in (source/'records.jsonl').read_text().splitlines()]
            assert len({r['record_id'] for r in old_rows}) == len(old_rows)
            for row in old_rows:
                if row['status'] != 'failed':
                    assert sha256_file(source/row['transform_file']) == row['transform_sha256']
                    assert sha256_file(source/'predictions'/(row['record_id']+'.npz')) == row['prediction_sha256']
            # Copy immutable fit artifacts, not ongoing audits or old manifests.
            artifacts = {f"bags/{r['task_id']}.json" for r in old_rows}
            for row in old_rows:
                if row['status'] != 'failed':
                    artifacts.update([row['transform_file'], 'predictions/'+row['record_id']+'.npz'])
            for name in artifacts:
                dest = root/name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source/name, dest)
            for name in ['records.jsonl', 'selected_parameters.json'] + [f'{d}_domain.json' for d in cc['domains']]:
                if (source/name).exists():
                    shutil.copy2(source/name, root/name)
            assert sha256_file(source/'records.jsonl') == source_digest == sha256_file(root/'records.jsonl')
            write_json(root/'import_provenance.json', dict(source=str(source), source_manifest_sha256=sha256_file(source/'run_manifest.json'), source_records_sha256=source_digest, imported_fits=len(old_rows), parity=parity, solver_changed=False))
        write_json(mp, identity)
        shutil.copy2(args.config, root/'config_snapshot.json')
        shutil.copy2(config['protocol_file'], root/'protocol_snapshot.md')
        for file in sources:
            dest = root/'source_snapshot'/file
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, dest)
    rows = [json.loads(v) for v in (root/'records.jsonl').read_text().splitlines()] if (root/'records.jsonl').exists() else []
    existing = {r['record_id'] for r in rows}
    selected_path = root/'selected_parameters.json'
    selected = json.loads(selected_path.read_text()) if selected_path.exists() else {}

    def run(tasks):
        for batch in run_tasks(tasks, root, config, cc, selected, existing, config['workers']):
            for row in batch:
                assert row['record_id'] not in existing
                store_record(root/'records.jsonl', rows, row)
                existing.add(row['record_id'])
                print(len(rows), row['domain'], row['method'], row['class_id'], row['seed'], row['noise'], row['split'], row['status'], flush=True)

    for domain in cc['domains']:
        dm = json.loads((core/f'{domain}_domain.json').read_text())
        path = root/f'{domain}_domain.json'
        if path.exists():
            assert json.loads(path.read_text()) == dm
        else:
            write_json(path, dm)
        run([(domain, cls, cc['validation_seed'], cc['validation_noise'], 'validate') for cls in dm['classes']])
        choice = selections(rows, domain, dm['classes'], config, cc['scales'])
        if domain in selected:
            assert selected[domain] == choice, 'Imported/resumed validation selection changed'
        selected[domain] = choice
        write_json(selected_path, selected)
        run([(domain, cls, seed, noise, 'test') for noise, seed, cls in itertools.product(cc['noise_levels'], cc['seeds'], dm['classes'])])
    print('Batch complete; independent audit still required', len(rows), flush=True)


if __name__ == '__main__':
    main()
