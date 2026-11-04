"""Parallel execution of the exact frozen task body, without solver edits."""
import ast
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from . import full_adaptation_wolfe_run as frozen


def task_id(domain, cls, seed, noise, split):
    return hashlib.sha256(json.dumps([domain, cls, seed, noise, split]).encode()).hexdigest()[:20]


def candidates(task, config, cc, selected):
    domain, cls, seed, noise, split = task
    result = []
    for adapter in config['adapters']:
        for head in frozen.HEADS:
            scales = cc['scales'] if split == 'validate' else [selected[domain][adapter]['heads'][head]['scale']]
            for scale in scales:
                key = dict(domain=domain, adapter=adapter, head=head, method=adapter+'+'+head,
                           class_id=cls, seed=seed, noise=noise, split=split, scale=scale)
                result.append(hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:20])
    return result


def initialize(root, config, cc, selected, existing):
    global _LIMITER, _NAMESPACE, _CACHE, _CAPTIONS, _DOMAINS
    _LIMITER = threadpool_limits(limits=1)
    with np.load(cc['cache']) as f:
        _CACHE = {k: f[k] for k in f.files}
    _CAPTIONS, _ = frozen.source_captions()
    _DOMAINS = {}
    # Compile the unchanged nested function from the frozen source. Its only
    # replaced side effects are the progress printer and the JSONL writer;
    # the parent serializes returned rows in the original canonical order.
    path = Path(frozen.__file__)
    tree = ast.parse(path.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    body = next(n for n in main.body if isinstance(n, ast.FunctionDef) and n.name == 'task')
    _NAMESPACE = dict(vars(frozen))
    _NAMESPACE.update(root=Path(root), config=config, cc=cc, selected=selected,
                      existing=set(existing), rows=[], log=None,
                      store_record=lambda log, rows, row: rows.append(row),
                      print=lambda *a, **k: None)
    exec(compile(ast.Module(body=[body], type_ignores=[]), str(path), 'exec'), _NAMESPACE)


def execute(task):
    domain, cls, seed, noise, split = task
    if domain not in _DOMAINS:
        cc = _NAMESPACE['cc']
        _DOMAINS[domain] = frozen.domain_cache(_CACHE, domain, captions=_CAPTIONS,
            n_bags=cc['n_bags'], bag_size=cc['bag_size'])[0]
    _NAMESPACE['rows'] = []
    _NAMESPACE['task'](domain, _DOMAINS[domain], cls, seed, noise, split)
    return _NAMESPACE['rows']


def run_tasks(tasks, root, config, cc, selected, existing, workers):
    tasks = [t for t in tasks if any(rid not in existing for rid in candidates(t, config, cc, selected))]
    if not tasks:
        return
    for name in ['bags', 'predictions', 'transforms']:
        (Path(root)/name).mkdir(exist_ok=True, parents=True)
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize,
            initargs=(str(root), config, cc, selected, existing)) as pool:
        # Ordered output preserves the validation summation/tie order while
        # independent event tasks compute concurrently in worker processes.
        yield from pool.map(execute, tasks, chunksize=1)
