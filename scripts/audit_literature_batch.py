"""Independent batch evidence audit; incomplete grid never passes delivery."""
import argparse,hashlib,itertools,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
from experiments.run import sha256_file
from experiments.core_run import source_captions
from experiments.paper_alignment import domain_cache
from experiments.data import make_task


def audit(root,allow_partial=False):
    root=Path(root);manifest=json.loads((root/'run_manifest.json').read_text());config=manifest['config'];core=Path(config['core_results'])
    cm=json.loads((core/'run_manifest.json').read_text());cc=cm['config']
    assert sha256_file(core/'run_manifest.json')==manifest['core_manifest_sha256']
    assert sha256_file(cc['cache'])==manifest['cache_sha256']
    assert sha256_file(config['protocol_file'])==manifest['protocol_sha256']
    assert sha256_file(root/'config_snapshot.json')==manifest['config_sha256']
    assert json.loads((root/'config_snapshot.json').read_text())==config
    assert sha256_file(root/'protocol_snapshot.md')==manifest['protocol_sha256']
    for f,h in manifest['source_hashes'].items():
        assert sha256_file(f)==h,f
        assert sha256_file(root/'source_snapshot'/f)==h,f
    rows=[json.loads(line) for line in (root/'records.jsonl').read_text().splitlines()]
    selected_path=root/'selected_parameters.json';selected=json.loads(selected_path.read_text()) if selected_path.exists() else {}
    with np.load(cc['cache']) as f:cache={k:f[k] for k in f.files}
    captions,_=source_captions();domains={};expected=set()
    with threadpool_limits(limits=1):
        for domain in cc['domains']:
            c,dm=domain_cache(cache,domain,captions=captions,n_bags=cc['n_bags'],bag_size=cc['bag_size']);domains[domain]=(c,dm)
            domain_path=root/f'{domain}_domain.json'
            if domain_path.exists():
                assert dm==json.loads(domain_path.read_text()),f'Domain identity differs: {domain}'
            else:
                assert allow_partial,f'Batch incomplete: missing domain artifact {domain_path}'
            for method,cls in itertools.product(config['methods'],dm['classes']):
                for scale in cc['scales']:expected.add((domain,method,cls,cc['validation_seed'],cc['validation_noise'],'validate',scale))
                for seed,noise in itertools.product(cc['seeds'],cc['noise_levels']):expected.add((domain,method,cls,seed,noise,'test',None))
    observed=set();seen_ids=set();bags={};checked_AP=0;statuses={};max_violation=0.
    for row in rows:
        rid=row['record_id'];assert rid not in seen_ids;seen_ids.add(rid)
        key=(row['domain'],row['method'],row['class_id'],row['seed'],row['noise'],row['split'],row['scale'] if row['split']=='validate' else None)
        assert key in expected and key not in observed,key;observed.add(key)
        identity={k:row[k] for k in ['domain','method','class_id','seed','noise','split','scale']}
        assert rid==hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:20]
        tk=(row['domain'],row['class_id'],row['seed'],row['noise'],row['split'])
        assert row['task_id']==hashlib.sha256(json.dumps(list(tk)).encode()).hexdigest()[:20]
        if tk not in bags:
            c,_=domains[row['domain']];data,bag=make_task(c,row['class_id'],seed=row['seed'],noise=row['noise'],n_bags=cc['n_bags'],bag_size=cc['bag_size'],evaluation_split=row['split'])
            stored=json.loads((root/'bags'/f"{row['task_id']}.json").read_text());assert stored==bag
            bags[tk]=(data.test_labels,bag['evaluation_video_ids'])
        statuses[row['status']]=statuses.get(row['status'],0)+1
        if row['split']=='test':assert row['scale']==selected[row['domain']][row['method']]
        if row['status']=='failed':continue
        pred=root/'predictions'/f'{rid}.npz';model=root/'models'/f'{rid}.npz'
        assert sha256_file(pred)==row['prediction_sha256'];assert sha256_file(model)==row['model_sha256']
        with np.load(pred) as p:
            assert np.isfinite(p['scores']).all()
            np.testing.assert_array_equal(p['labels'],bags[tk][0]);np.testing.assert_array_equal(p['video_ids'],bags[tk][1])
            ap=average_precision_score(p['labels']>0,p['scores']);assert abs(ap-row['average_precision'])<1e-12
        checked_AP+=1;v=row['diagnostics'].get('max_constraint_violation',0.);max_violation=max(max_violation,v);assert v<=1e-6
        if row['status']=='ok':assert row['diagnostics']['stop_reason'] not in ['max_iter','line_search_failed']
    # Independently recompute complete-class validation macro AP.
    for domain,choices in selected.items():
        classes=domains[domain][1]['classes']
        for method,chosen in choices.items():
            options=[]
            for scale in cc['scales']:
                sub=[r for r in rows if r['domain']==domain and r['method']==method and r['split']=='validate' and r['scale']==scale]
                if len(sub)==len(classes) and {r['class_id'] for r in sub}==set(classes) and all(r['status']=='ok' for r in sub):options.append((scale,float(np.mean([r['average_precision'] for r in sub]))))
            assert options and chosen==max(options,key=lambda p:p[1])[0]
    missing=len(expected-observed);bad_test=sum(r['split']=='test' and r['status']!='ok' for r in rows)
    complete=not missing and not bad_test and set(selected)==set(cc['domains'])
    result=dict(status='grid_and_predictions_passed' if complete else 'partial_not_delivery',expected_fits=len(expected),observed_fits=len(rows),missing_fits=missing,status_counts=statuses,APs_recomputed=checked_AP,bag_tasks_checked=len(bags),max_constraint_violation=max_violation,failed_or_incomplete_test_fits=bad_test,scope=config['scope'],model_replay_audit='separate requirement; DASVM requires supplemental replayable classifier')
    if not allow_partial:assert complete,result
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--allow-partial',action='store_true');args=p.parse_args()
    r=audit(args.root,args.allow_partial);output=args.root/('partial_audit.json' if args.allow_partial else 'grid_prediction_audit.json');output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
