"""Frozen concept event evaluation; domains wait for complete score caches."""
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
from .concept_classification import concept_event_problem, event_rows
from .concept_event_baseline import fit_concept_event
from .run import sha256_file, write_json, store_record


def select_scale(rows, domain, method, classes, scales):
    options = []
    for scale in scales:
        sub = [r for r in rows if (r['domain'],r['method'],r['split'],r['scale']) == (domain,method,'validate',scale)]
        ids = [r['class_id'] for r in sub]
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate validation class')
        if set(ids) == set(classes) and all(r['status'] == 'ok' for r in sub):
            options.append((scale, float(np.mean([r['average_precision'] for r in sub]))))
    if not options:
        raise RuntimeError('No complete validation candidate: '+domain+'/'+method)
    return max(options, key=lambda r:r[1])[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/concept_events.json'))
    parser.add_argument('--output', type=Path, default=Path('results/concept_events'))
    parser.add_argument('--phase', choices=['tune','evaluate','all'], default='all')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    assert config['methods'] == ['Classeme','MIMLSVM']
    core, representations = Path(config['core_results']), Path(config['representations'])
    cm = json.loads((core/'run_manifest.json').read_text());cc = cm['config']
    rm = json.loads((representations/'run_manifest.json').read_text())
    assert rm['core_manifest_sha256'] == sha256_file(core/'run_manifest.json')
    assert rm['cache_sha256'] == sha256_file(cc['cache'])
    assert config['domains'] == cc['domains'] and config['concept_weight'] == 1.
    sources = set(cm['source_hashes']) | set(rm['source_hashes']) | {'experiments/concept_event_run.py','experiments/concept_event_baseline.py','experiments/concept_classification.py'}
    for m in [cm,rm]:
        for f,digest in m['source_hashes'].items():assert sha256_file(f) == digest,f
    root = args.output;root.mkdir(parents=True,exist_ok=True)
    identity = dict(config=config, config_sha256=sha256_file(args.config), core_manifest_sha256=sha256_file(core/'run_manifest.json'), representation_manifest_sha256=sha256_file(representations/'run_manifest.json'), cache_sha256=sha256_file(cc['cache']), protocol_sha256=sha256_file(config['protocol_file']), source_hashes={f:sha256_file(f) for f in sorted(sources)})
    mp = root/'run_manifest.json'
    if mp.exists():assert json.loads(mp.read_text()) == identity,'Frozen event identity changed'
    else:
        write_json(mp,identity)
        shutil.copy2(args.config,root/'config_snapshot.json');shutil.copy2(config['protocol_file'],root/'protocol_snapshot.md')
        for f in sources:
            dest=root/'source_snapshot'/f;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
    for name in ['predictions','models','bags','kernels']:(root/name).mkdir(exist_ok=True)
    with np.load(cc['cache']) as f:cache={k:f[k] for k in f.files}
    captions,_=source_captions()
    log=root/'records.jsonl';rows=[json.loads(s) for s in log.read_text().splitlines()] if log.exists() else []
    existing={r['record_id'] for r in rows};sp=root/'selected_parameters.json'
    selected=json.loads(sp.read_text()) if sp.exists() else {};pending=[]
    with threadpool_limits(limits=1):
        for domain in config['domains']:
            if any(not (representations/f'{domain}_{m}_complete.json').exists() for m in config['methods']):
                pending.append(domain);print('Pending representation domain',domain,flush=True);continue
            c,dm=domain_cache(cache,domain,captions=captions,n_bags=cc['n_bags'],bag_size=cc['bag_size'])
            assert dm==json.loads((core/f'{domain}_domain.json').read_text())
            write_json(root/f'{domain}_domain.json',dm)
            scores={};bindings={}
            for method in config['methods']:
                prefix=f'{domain}_{method}';complete=representations/f'{prefix}_complete.json'
                report=json.loads(complete.read_text());model=representations/f'{prefix}.joblib';score=representations/f'{prefix}_scores.npz'
                assert report['manifest_sha256']==sha256_file(representations/'run_manifest.json')
                assert sha256_file(model)==report['model_sha256'] and sha256_file(score)==report['scores_sha256']
                bindings[method]=dict(model_sha256=report['model_sha256'],scores_sha256=report['scores_sha256'],complete_sha256=sha256_file(complete))
                with np.load(score) as f:
                    np.testing.assert_array_equal(f['video_ids'],c['video_ids']);np.testing.assert_array_equal(f['splits'],c['splits'])
                    np.testing.assert_array_equal(f['represented_mask'],np.isin(c['splits'],['train','validate','test']))
                    scores[method]=f['scores']
                    assert scores[method].shape==(len(c['video_ids']),2000) and np.isfinite(scores[method]).all()
            binding_path=root/f'{domain}_representations.json'
            if binding_path.exists():assert json.loads(binding_path.read_text())==bindings
            else:write_json(binding_path,bindings)

            def task(cls,seed,noise,split):
                tid=hashlib.sha256(json.dumps([domain,cls,seed,noise,split]).encode()).hexdigest()[:20]
                data,bag=make_task(c,cls,seed=seed,noise=noise,n_bags=cc['n_bags'],bag_size=cc['bag_size'],evaluation_split=split)
                bp=root/'bags'/f'{tid}.json'
                if bp.exists():assert json.loads(bp.read_text())==bag
                else:write_json(bp,bag)
                for method in config['methods']:
                    candidates=[]
                    for scale in cc['scales'] if split=='validate' else [selected[domain][method]]:
                        key=dict(domain=domain,method=method,class_id=cls,seed=seed,noise=noise,split=split,scale=scale)
                        rid=hashlib.sha256(json.dumps(key,sort_keys=True).encode()).hexdigest()[:20]
                        if rid not in existing:candidates.append((key,rid))
                    if not candidates:continue
                    start=time.perf_counter();problem_error=None
                    try:
                        p=concept_event_problem(c['visual'],scores[method],c['video_ids'],bag,concept_weight=1.)
                        kp=root/'kernels'/f'{tid}_{method}.npz'
                        if kp.exists():
                            with np.load(kp) as f:
                                np.testing.assert_allclose(f['K'],p['K'],rtol=1e-10,atol=1e-12);np.testing.assert_allclose(f['cross'],p['test_kernel'],rtol=1e-10,atol=1e-12)
                        else:np.savez_compressed(kp,K=p['K'],cross=p['test_kernel'],kernel_gamma=np.array(p['kernel_gamma']))
                    except Exception as error:problem_error=str(error)
                    for key,rid in candidates:
                        row=dict(key,record_id=rid,task_id=tid)
                        try:
                            if problem_error:raise RuntimeError(problem_error)
                            pred,diag,arrays=fit_concept_event(p,scale=key['scale'])
                            pp=root/'predictions'/f'{rid}.npz';model=root/'models'/f'{rid}.npz'
                            np.savez_compressed(pp,scores=pred,labels=data.test_labels,video_ids=bag['evaluation_video_ids'])
                            np.savez_compressed(model,**arrays)
                            row.update(status='ok',average_precision=float(average_precision_score(data.test_labels>0,pred)),diagnostics=diag,prediction_sha256=sha256_file(pp),model_sha256=sha256_file(model),kernel_file=str(kp.relative_to(root)),kernel_sha256=sha256_file(kp))
                        except Exception as error:row.update(status='failed',error=str(error),traceback=traceback.format_exc())
                        row['seconds_since_problem_start']=time.perf_counter()-start
                        store_record(log,rows,row);existing.add(rid)
                        print(len(rows),domain,method,cls,seed,noise,split,row['status'],flush=True)
            if args.phase in ['tune','all']:
                for cls in dm['classes']:task(cls,cc['validation_seed'],cc['validation_noise'],'validate')
                selected[domain]={m:select_scale(rows,domain,m,dm['classes'],cc['scales']) for m in config['methods']};write_json(sp,selected)
            if args.phase in ['evaluate','all']:
                assert selected[domain]=={m:select_scale(rows,domain,m,dm['classes'],cc['scales']) for m in config['methods']}
                for noise in cc['noise_levels']:
                    for seed in cc['seeds']:
                        for cls in dm['classes']:task(cls,seed,noise,'test')
    write_json(root/'execution_status.json',dict(records=len(rows),pending_representation_domains=pending,independent_audit_required=True))
    print('Snapshot finished; independent audit required; pending:',pending,flush=True)


if __name__=='__main__':main()
