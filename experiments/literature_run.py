"""Frozen partial-method batch; full-baseline completion remains separate."""
import argparse,hashlib,json,shutil,time,traceback,importlib.metadata
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
from .core_run import source_captions
from .paper_alignment import domain_cache
from .data import make_task
from .models import prepare_problem
from .literature_dispatch import fit_literature,METHODS
from .run import sha256_file,write_json,store_record


def select_scale(records,domain,method,classes,scales):
    options=[]
    for scale in scales:
        subset=[r for r in records if r['domain']==domain and r['method']==method and r['split']=='validate' and r['scale']==scale]
        if len(subset)!=len(classes) or {r['class_id'] for r in subset}!=set(classes) or any(r['status']!='ok' for r in subset):continue
        options.append((scale,float(np.mean([r['average_precision'] for r in subset]))))
    if not options:raise RuntimeError('No complete successful validation candidate: '+domain+'/'+method)
    return max(options,key=lambda v:v[1])[0]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/literature.json'))
    parser.add_argument('--output',type=Path,default=Path('results/literature'))
    parser.add_argument('--phase',choices=['tune','evaluate','all'],default='all')
    args=parser.parse_args();config=json.loads(args.config.read_text());root=args.output;root.mkdir(parents=True,exist_ok=True)
    if any(m not in METHODS for m in config['methods']):raise ValueError('Unverified method requested')
    core=Path(config['core_results']);manifest=json.loads((core/'run_manifest.json').read_text());cc=manifest['config']
    for f,digest in manifest['source_hashes'].items():
        if sha256_file(f)!=digest:raise ValueError('Core source changed: '+f)
    # Explicit source dependency closure: unrelated new-method edits are safe.
    sources=set(manifest['source_hashes'])|set(config['additional_sources'])|{'experiments/literature_run.py'}
    identity=dict(config=config,config_sha256=sha256_file(args.config),core_manifest_sha256=sha256_file(core/'run_manifest.json'),cache_sha256=sha256_file(cc['cache']),protocol_sha256=sha256_file(config['protocol_file']),source_hashes={f:sha256_file(f) for f in sorted(sources)},versions={n:importlib.metadata.version(n) for n in ['numpy','scipy','scikit-learn','cvxpy','clarabel','osqp']})
    mp=root/'run_manifest.json'
    if mp.exists():
        if json.loads(mp.read_text())!=identity:raise ValueError('Frozen run identity changed; use a new output directory')
    else:
        write_json(mp,identity);shutil.copy2(args.config,root/'config_snapshot.json');shutil.copy2(config['protocol_file'],root/'protocol_snapshot.md')
        for f in sources:
            target=root/'source_snapshot'/f;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,target)
    for directory in ['predictions','models','bags']:(root/directory).mkdir(exist_ok=True)
    with np.load(cc['cache']) as f:cache={k:f[k] for k in f.files}
    captions,_=source_captions();log=root/'records.jsonl';records=[json.loads(s) for s in log.read_text().splitlines()] if log.exists() else []
    existing={r['record_id'] for r in records};sp=root/'selected_parameters.json';selected=json.loads(sp.read_text()) if sp.exists() else {}
    def task(domain,c,cls,seed,noise,split,scales):
        data,bag=make_task(c,cls,seed=seed,noise=noise,n_bags=cc['n_bags'],bag_size=cc['bag_size'],evaluation_split=split);problem=prepare_problem(data)
        tid=hashlib.sha256(json.dumps([domain,cls,seed,noise,split]).encode()).hexdigest()[:20]
        bp=root/'bags'/f'{tid}.json'
        if bp.exists():
            if json.loads(bp.read_text())!=bag:raise ValueError('Task bag identity changed')
        else:write_json(bp,bag)
        for method in config['methods']:
            candidates=scales if scales is not None else [selected[domain][method]]
            for scale in candidates:
                key=dict(domain=domain,method=method,class_id=cls,seed=seed,noise=noise,split=split,scale=scale)
                rid=hashlib.sha256(json.dumps(key,sort_keys=True).encode()).hexdigest()[:20]
                if rid in existing:continue
                row=dict(key,record_id=rid,task_id=tid);start=time.perf_counter()
                try:
                    scores,diag,arrays=fit_literature(method,data,problem,scale=scale)
                    np.savez_compressed(root/'predictions'/f'{rid}.npz',scores=scores,labels=data.test_labels,video_ids=bag['evaluation_video_ids'])
                    np.savez_compressed(root/'models'/f'{rid}.npz',**arrays)
                    # Capped iterative solvers are retained but not selected.
                    status='incomplete' if diag['stop_reason'] in ['max_iter','line_search_failed'] else 'ok'
                    row.update(status=status,average_precision=float(average_precision_score(data.test_labels>0,scores)),diagnostics=diag,prediction_sha256=sha256_file(root/'predictions'/f'{rid}.npz'),model_sha256=sha256_file(root/'models'/f'{rid}.npz'))
                except Exception as error:row.update(status='failed',error=str(error),traceback=traceback.format_exc())
                row['seconds']=time.perf_counter()-start;store_record(log,records,row);existing.add(rid)
                print(len(records),domain,method,cls,seed,noise,split,row['status'],flush=True)
    with threadpool_limits(limits=1):
        for domain in cc['domains']:
            c,dm=domain_cache(cache,domain,captions=captions,n_bags=cc['n_bags'],bag_size=cc['bag_size'])
            if dm!=json.loads((core/f'{domain}_domain.json').read_text()):raise ValueError('Core domain identity differs')
            write_json(root/f'{domain}_domain.json',dm)
            if args.phase in ['tune','all']:
                for cls in dm['classes']:task(domain,c,cls,cc['validation_seed'],cc['validation_noise'],'validate',cc['scales'])
                selected[domain]={m:select_scale(records,domain,m,dm['classes'],cc['scales']) for m in config['methods']};write_json(sp,selected)
            if args.phase in ['evaluate','all']:
                if domain not in selected:raise ValueError('Validation selection missing')
                # Recompute selection before any test fit, including resumes.
                expected={m:select_scale(records,domain,m,dm['classes'],cc['scales']) for m in config['methods']}
                if selected[domain]!=expected:raise ValueError('Saved selection differs from validation')
                for noise in cc['noise_levels']:
                    for seed in cc['seeds']:
                        for cls in dm['classes']:task(domain,c,cls,seed,noise,'test',None)
    print('Batch finished; separate evidence audit required. Records:',len(records),flush=True)


if __name__=='__main__':main()
