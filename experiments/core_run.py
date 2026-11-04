"""Frozen protocol runner: tune on validation, then all-family five-seed tests."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import importlib.metadata
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import time
import traceback
from zipfile import ZipFile
import numpy as np
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
from .paper_alignment import domain_cache
from .data import make_task
from .models import prepare_problem
from .core_models import fit,parameters,family_of
from .run import sha256_file,write_json,store_record


def source_captions():
    archive=Path('data/downloads/msrvtt.zip')
    expected='bb8cc603f9cfd0543dfa03bc88a4dae887f97ffcd21a59fc97aecc2296311828'
    if sha256_file(archive)!=expected:raise ValueError('Caption archive changed')
    with ZipFile(archive) as z:annotation=json.loads(z.read('msrvtt/Annotations/MSRVTT_data.json'))
    grouped=defaultdict(list)
    for row in annotation['sentences']:grouped[row['video_id']].append(row['caption'])
    return {k:' '.join(v) for k,v in grouped.items()},expected


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/core.json'))
    parser.add_argument('--output',type=Path,default=Path('results/core'))
    parser.add_argument('--phase',choices=['all','tune','evaluate'],default='all')
    parser.add_argument('--retry-failed',action='store_true')
    args=parser.parse_args(argv);config=json.loads(args.config.read_text());root=args.output;root.mkdir(exist_ok=True,parents=True)
    captions,archive_sha=source_captions()
    path=root/'run_manifest.json'
    source_files=[Path(p) for p in json.loads(path.read_text())['source_hashes']] if path.exists() else [p for directory in ['mil_pi','experiments'] for p in sorted(Path(directory).glob('*.py'))]
    identity=dict(config=config,config_sha256=sha256_file(args.config),cache_sha256=sha256_file(config['cache']),
        caption_archive_sha256=archive_sha,protocol_sha256=sha256_file('docs/core_protocol.md'),
        source_hashes={str(p):sha256_file(p) for p in source_files},
        versions={name:importlib.metadata.version(name) for name in ['numpy','scipy','scikit-learn','osqp','cvxpy','clarabel','scs']})
    path=root/'run_manifest.json'
    if path.exists():
        if json.loads(path.read_text())!=identity:raise ValueError('Run identity changed; use a new result directory')
    else:
        write_json(path,identity)
        for filename in identity['source_hashes']:
            q=root/'source_snapshot'/filename;q.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(filename,q)
        shutil.copy2(args.config,root/'config_snapshot.json');shutil.copy2('docs/core_protocol.md',root/'protocol_snapshot.md')
    with np.load(config['cache']) as stored:cache={k:stored[k] for k in stored.files}
    log=root/'records.jsonl';rows=[json.loads(s) for s in log.read_text().splitlines()] if log.exists() else []
    existing={r['record_id']:r for r in rows}
    for sub in ['predictions','bags']: (root/sub).mkdir(exist_ok=True)
    problems={};problem_lock=Lock();record_lock=Lock()
    def run(domain,c,method,cls,seed,noise,split,p,candidate):
        key=dict(domain=domain,method=method,class_id=cls,seed=seed,noise=noise,split=split,parameters=p,candidate=candidate)
        rid=hashlib.sha256(json.dumps(key,sort_keys=True).encode()).hexdigest()[:20]
        if rid in existing and (existing[rid]['status']=='ok' or not args.retry_failed):return existing[rid]
        record=dict(**key,record_id=rid);started=time.perf_counter()
        task=(domain,cls,seed,noise,split)
        try:
            with problem_lock:
                if task not in problems:
                    data,bag=make_task(c,cls,seed=seed,noise=noise,n_bags=config['n_bags'],bag_size=config['bag_size'],evaluation_split=split)
                    problem=prepare_problem(data);text=np.concatenate(data.positive_text_bags+data.negative_text_bags,axis=1)
                    problem['full_tK']=text.T@text
                    if len(problems)>=20:problems.clear()
                    problems[task]=(data,bag,problem)
                data,bag,problem=problems[task]
            scores,diag=fit(method,problem,p,seed=seed)
            ap=float(average_precision_score(data.test_labels>0,scores))
            record.update(status='ok',average_precision=ap,**diag)
            file=root/'predictions'/f'{rid}.npz'
            with file.with_suffix('.tmp').open('wb') as out:np.savez_compressed(out,scores=scores,labels=data.test_labels,video_ids=bag['evaluation_video_ids'])
            file.with_suffix('.tmp').replace(file)
            taskid=hashlib.sha256(json.dumps(task).encode()).hexdigest()[:20]
            with record_lock:write_json(root/'bags'/f'{taskid}.json',dict(domain=domain,task_id=taskid,**bag))
        except Exception as error:
            record.update(status='failed',error=str(error),traceback=traceback.format_exc(),seconds=time.perf_counter()-started)
        with record_lock:
            store_record(log,rows,record);existing[rid]=record
        print(f'{len(rows):5d} {domain} {split} {method} class={cls} seed={seed} noise={noise}: '+
              (f'{record["average_precision"]:.5f} {record["stop_reason"]}' if record['status']=='ok' else record['error']),flush=True)
        return record
    selected_path=root/'selected_parameters.json';candidate_path=root/'validation_candidates.json'
    selected=json.loads(selected_path.read_text()) if selected_path.exists() else {}
    validation=json.loads(candidate_path.read_text()) if candidate_path.exists() else {}
    with threadpool_limits(limits=1),ThreadPoolExecutor(max_workers=config['workers']) as pool:
        for domain in config['domains']:
            c,dm=domain_cache(cache,domain,n_bags=config['n_bags'],bag_size=config['bag_size'],captions=captions)
            write_json(root/f'{domain}_domain.json',dm);classes=dm['classes']
            if args.phase in {'all','tune'}:
                selected[domain]={};validation[domain]={}
                for method in config['base_methods']:
                    options=[]
                    for scale in config['scales']:
                        p=parameters(method,scale)
                        fits=list(pool.map(lambda cls:run(domain,c,method,cls,config['validation_seed'],config['validation_noise'],'validate',p,scale),classes))
                        complete=all(r['status']=='ok' for r in fits)
                        option=dict(scale=scale,complete=complete,failed_fits=sum(r['status']!='ok' for r in fits))
                        if complete:option['mAP']=float(np.mean([r['average_precision'] for r in fits]))
                        options.append(option)
                    validation[domain][method]=options;write_json(candidate_path,validation)
                    good=[r for r in options if r['complete']]
                    if not good:raise RuntimeError(f'No complete validation candidate for {domain}/{method}; inspect failed records')
                    best=max(good,key=lambda r:r['mAP']);selected[domain][method]=parameters(method,best['scale'])
                    write_json(selected_path,selected)
                for method in config['dependent_controls']:
                    selected[domain][method]=selected[domain][family_of(method)+'-PI'].copy()
                write_json(selected_path,selected)
            if args.phase in {'all','evaluate'}:
                for noise in config['noise_levels']:
                    for seed in config['seeds']:
                        for cls in classes:
                            list(pool.map(lambda method:run(domain,c,method,cls,seed,noise,'test',selected[domain][method],'selected_on_validation'),config['base_methods']+config['dependent_controls']))
    summary=[]
    for domain in config['domains']:
        for noise in config['noise_levels']:
            for method in config['base_methods']+config['dependent_controls']:
                maps=[];failures=0
                for seed in config['seeds']:
                    subset=[r for r in rows if r['domain']==domain and r['noise']==noise and r['method']==method and r['split']=='test' and r['seed']==seed]
                    failures+=sum(r['status']!='ok' for r in subset)
                    if subset and all(r['status']=='ok' for r in subset):maps.append(float(np.mean([r['average_precision'] for r in subset])))
                summary.append(dict(domain=domain,noise=noise,method=method,seed_mAPs=maps,failed_fits=failures,
                    mean_mAP=float(np.mean(maps)) if maps else None,std_mAP=float(np.std(maps,ddof=1)) if len(maps)>1 else None))
    write_json(root/'summary.json',summary)
    print('Saved',root/'summary.json',len(rows),'fits',flush=True)

if __name__=='__main__':main()
