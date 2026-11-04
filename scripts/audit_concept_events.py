"""Independent concept event grid/AP/selection audit and model replay."""
import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
from experiments.core_run import source_captions
from experiments.paper_alignment import domain_cache
from experiments.data import make_task
from experiments.concept_classification import concept_event_problem
from experiments.run import sha256_file


def audit(root,partial=False):
    root=Path(root);manifest=json.loads((root/'run_manifest.json').read_text());config=manifest['config']
    core=Path(config['core_results']);rep=Path(config['representations'])
    assert sha256_file(core/'run_manifest.json')==manifest['core_manifest_sha256']
    assert sha256_file(rep/'run_manifest.json')==manifest['representation_manifest_sha256']
    cc=json.loads((core/'run_manifest.json').read_text())['config']
    assert sha256_file(cc['cache'])==manifest['cache_sha256']
    assert sha256_file(root/'config_snapshot.json')==manifest['config_sha256']
    assert json.loads((root/'config_snapshot.json').read_text())==config
    assert sha256_file(config['protocol_file'])==manifest['protocol_sha256']
    assert sha256_file(root/'protocol_snapshot.md')==manifest['protocol_sha256']
    for file,digest in manifest['source_hashes'].items():
        assert sha256_file(file)==digest,file
        assert sha256_file(root/'source_snapshot'/file)==digest,file
    with np.load(cc['cache']) as f:cache={k:f[k] for k in f.files}
    captions,_=source_captions();expected=set();domains={};score_cache={}
    path=root/'records.jsonl';rows=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    sp=root/'selected_parameters.json';selected=json.loads(sp.read_text()) if sp.exists() else {}
    with threadpool_limits(limits=1):
        for domain in config['domains']:
            c,dm=domain_cache(cache,domain,captions=captions,n_bags=cc['n_bags'],bag_size=cc['bag_size'])
            domains[domain]=(c,dm)
            for method,cls in itertools.product(config['methods'],dm['classes']):
                for scale in cc['scales']:expected.add((domain,method,cls,cc['validation_seed'],cc['validation_noise'],'validate',scale))
                for seed,noise in itertools.product(cc['seeds'],cc['noise_levels']):expected.add((domain,method,cls,seed,noise,'test',None))
            dp=root/f'{domain}_domain.json'
            if not dp.exists():
                assert partial,f'Missing domain {domain}'
                continue
            assert json.loads(dp.read_text())==dm
            binding=json.loads((root/f'{domain}_representations.json').read_text())
            score_cache[domain]={}
            for method in config['methods']:
                prefix=f'{domain}_{method}'
                assert sha256_file(rep/f'{prefix}_complete.json')==binding[method]['complete_sha256']
                complete=json.loads((rep/f'{prefix}_complete.json').read_text())
                assert complete['manifest_sha256']==manifest['representation_manifest_sha256']
                assert complete['model_sha256']==binding[method]['model_sha256']==sha256_file(rep/f'{prefix}.joblib')
                assert complete['scores_sha256']==binding[method]['scores_sha256']==sha256_file(rep/f'{prefix}_scores.npz')
                with np.load(rep/f'{prefix}_scores.npz') as f:
                    np.testing.assert_array_equal(f['video_ids'],c['video_ids']);np.testing.assert_array_equal(f['splits'],c['splits'])
                    np.testing.assert_array_equal(f['represented_mask'],np.isin(c['splits'],['train','validate','test']))
                    score_cache[domain][method]=f['scores']
        seen=set();tasks={};kernels=set();statuses={};max_error=0.;max_violation=0.
        for r in rows:
            key=(r['domain'],r['method'],r['class_id'],r['seed'],r['noise'],r['split'],r['scale'] if r['split']=='validate' else None)
            assert key in expected and key not in seen,key;seen.add(key)
            identity={k:r[k] for k in ['domain','method','class_id','seed','noise','split','scale']}
            rid=r['record_id'];assert rid==hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:20]
            tk=(r['domain'],r['class_id'],r['seed'],r['noise'],r['split'])
            assert r['task_id']==hashlib.sha256(json.dumps(list(tk)).encode()).hexdigest()[:20]
            if tk not in tasks:
                c=domains[r['domain']][0]
                data,bag=make_task(c,r['class_id'],seed=r['seed'],noise=r['noise'],n_bags=cc['n_bags'],bag_size=cc['bag_size'],evaluation_split=r['split'])
                assert json.loads((root/'bags'/(r['task_id']+'.json')).read_text())==bag
                tasks[tk]=(data.test_labels,bag)
            labels,bag=tasks[tk]
            statuses[r['status']]=statuses.get(r['status'],0)+1
            if r['split']=='test':assert r['scale']==selected[r['domain']][r['method']]
            if r['status']!='ok':continue
            diag=r['diagnostics'];v=diag['max_constraint_violation'];assert v<=1e-6;max_violation=max(max_violation,v)
            assert diag['positive_cost']==100*r['scale'] and diag['negative_cost']==1.
            assert diag['target_labels_used'] is False
            assert diag['stop_reason'] not in ['max_iter','line_search_failed']
            kernel=r['kernel_file'];assert kernel==f"kernels/{r['task_id']}_{r['method']}.npz"
            assert sha256_file(root/kernel)==r['kernel_sha256']
            with np.load(root/kernel) as k:
                if kernel not in kernels:
                    c=domains[r['domain']][0]
                    p=concept_event_problem(c['visual'],score_cache[r['domain']][r['method']],c['video_ids'],bag,concept_weight=1.)
                    np.testing.assert_allclose(k['K'],p['K'],rtol=1e-10,atol=1e-12)
                    np.testing.assert_allclose(k['cross'],p['test_kernel'],rtol=1e-10,atol=1e-12)
                    assert k['kernel_gamma'].item()==p['kernel_gamma'];kernels.add(kernel)
                model=root/'models'/(rid+'.npz');pred=root/'predictions'/(rid+'.npz')
                assert sha256_file(model)==r['model_sha256'] and sha256_file(pred)==r['prediction_sha256']
                with np.load(model) as m,np.load(pred) as saved:
                    actual=k['cross']@m['coef']+m['bias']
                    np.testing.assert_allclose(actual,saved['scores'],rtol=1e-10,atol=1e-10)
                    max_error=max(max_error,float(np.max(np.abs(actual-saved['scores']))))
                    np.testing.assert_array_equal(saved['labels'],labels)
                    np.testing.assert_array_equal(saved['video_ids'],bag['evaluation_video_ids'])
                    assert abs(average_precision_score(labels>0,saved['scores'])-r['average_precision'])<1e-12
        for domain,choices in selected.items():
            classes=set(domains[domain][1]['classes'])
            for method,chosen in choices.items():
                candidates=[]
                for scale in cc['scales']:
                    sub=[r for r in rows if (r['domain'],r['method'],r['split'],r['scale'])==(domain,method,'validate',scale)]
                    if len(sub)==len(classes) and {r['class_id'] for r in sub}==classes and all(r['status']=='ok' for r in sub):candidates.append((scale,float(np.mean([r['average_precision'] for r in sub]))))
                assert candidates and chosen==max(candidates,key=lambda p:p[1])[0]
        bad_test=sum(r['split']=='test' and r['status']!='ok' for r in rows)
        complete=seen==expected and not bad_test and set(selected)==set(config['domains']) and all(set(v)==set(config['methods']) for v in selected.values())
        result=dict(status='grid_AP_selection_kernels_and_models_passed' if complete else 'partial_not_delivery',expected_fits=len(expected),observed_fits=len(rows),missing_fits=len(expected-seen),status_counts=statuses,bag_tasks_checked=len(tasks),kernels_recomputed=len(kernels),max_model_prediction_error=max_error,max_constraint_violation=max_violation,failed_test_fits=bad_test,scope='Event grid/AP, representation identity, kernel reconstruction and serialized model forward replay; representation training audit remains separate',script_sha256=sha256_file(__file__))
        if not partial:assert complete,result
        return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--allow-partial',action='store_true');args=p.parse_args()
    result=audit(args.root,args.allow_partial)
    (args.root/('partial_audit.json' if args.allow_partial else 'delivery_audit.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
