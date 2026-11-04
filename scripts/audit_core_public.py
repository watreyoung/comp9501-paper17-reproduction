"""Audit the frozen protocol, including failed validation candidates and all seeds."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=Path('results/core'))
    parser.add_argument('--validation-only',action='store_true')
    args=parser.parse_args();root=args.results
    manifest=json.loads((root/'run_manifest.json').read_text());c=manifest['config']
    assert digest(root/'config_snapshot.json')==manifest['config_sha256']
    assert digest(root/'protocol_snapshot.md')==manifest['protocol_sha256']
    assert digest(c['cache'])==manifest['cache_sha256']
    for p,h in manifest['source_hashes'].items():
        assert digest(root/'source_snapshot'/p)==h,p
        assert digest(p)==h,p
    assert digest('data/downloads/msrvtt.zip')==manifest['caption_archive_sha256']
    with np.load(c['cache']) as f:cache={k:f[k] for k in f.files}
    groups={s:set(cache['video_ids'][cache['splits']==s]) for s in ['train','validate','test']}
    labels=dict(zip(cache['video_ids'],cache['labels']))
    domains={d:json.loads((root/f'{d}_domain.json').read_text()) for d in c['domains']}
    for d,m in domains.items():
        ids={s:set(v) for s,v in m['split_ids'].items()}
        assert not(ids['train']&ids['validate'] or ids['train']&ids['test'] or ids['validate']&ids['test'])
        assert all(ids[s]<=groups[s] for s in ids)
        classes=[i for i in range(20) if sum(labels[v]==i for v in ids['train'])>=25 and sum(labels[v]==i for v in ids['validate'])>=3]
        assert classes==m['classes']
        if d=='visual_cluster':assert set(m['text_vocabulary_fit_ids'])==ids['train']
    rows=[json.loads(s) for s in (root/'records.jsonl').read_text().splitlines()]
    assert len(rows)==len({r['record_id'] for r in rows})
    validation=[r for r in rows if r['split']=='validate'];tests=[r for r in rows if r['split']=='test']
    expected_val={(d,method,cls,scale) for d,m in domains.items() for method in c['base_methods'] for cls in m['classes'] for scale in c['scales']}
    actual_val={(r['domain'],r['method'],r['class_id'],r['candidate']) for r in validation}
    assert len(validation)==len(expected_val) and actual_val==expected_val
    selected=json.loads((root/'selected_parameters.json').read_text());options=json.loads((root/'validation_candidates.json').read_text())
    def family(method):
        return next((f for f in c['families'] if f in method),method)
    for d in domains:
        for method in c['base_methods']:
            candidates=[]
            for scale in c['scales']:
                subset=sorted([r for r in validation if r['domain']==d and r['method']==method and r['candidate']==scale], key=lambda r: domains[d]['classes'].index(r['class_id']))
                found=next(x for x in options[d][method] if x['scale']==scale)
                good=all(r['status']=='ok' for r in subset)
                assert found['complete']==good
                assert found['failed_fits']==sum(r['status']!='ok' for r in subset)
                if good:
                    mean=float(np.mean([r['average_precision'] for r in subset]));assert abs(found['mAP']-mean)<1e-12
                    candidates.append((scale,mean))
            assert candidates
            best=max(candidates,key=lambda x:x[1])[0]
            base=.01 if family(method) in ['mi-SVM','MIL-CPB'] else 100.
            assert selected[d][method]['C1']==base*best
        for method in c['dependent_controls']:assert selected[d][method]==selected[d][family(method)+'-PI']
    if not args.validation_only:
        expected_test={(d,method,cls,seed,noise) for d,m in domains.items() for method in c['base_methods']+c['dependent_controls'] for cls in m['classes'] for seed in c['seeds'] for noise in c['noise_levels']}
        actual_test={(r['domain'],r['method'],r['class_id'],r['seed'],r['noise']) for r in tests}
        assert len(tests)==len(expected_test) and actual_test==expected_test
        assert all(r['status']=='ok' for r in tests),'test failures must be resolved or reported as incomplete'
    ap_count=0;cone_count=0;fallback_count=0;limits=[];max_violation=0.
    for r in rows:
        if r['status']!='ok':
            assert r['split']=='validate' and r['error'];continue
        if r['split']=='test':assert r['parameters']==selected[r['domain']][r['method']]
        with np.load(root/'predictions'/f'{r["record_id"]}.npz') as p:
            assert set(p['video_ids'])==set(domains[r['domain']]['split_ids'][r['split']])
            assert len(p['video_ids'])==len(set(p['video_ids']))
            expected=np.array([1 if labels[v]==r['class_id'] else -1 for v in p['video_ids']])
            np.testing.assert_array_equal(p['labels'],expected)
            assert np.isfinite(p['scores']).all()
            assert abs(average_precision_score(expected>0,p['scores'])-r['average_precision'])<1e-12
            ap_count+=1
        assert all(q['success'] and q['constraint_violation']<=1e-6 for q in r['qp_diagnostics'])
        for q in r['model_diagnostics'].get('conic_diagnostics',[]):
            assert q['status']=='optimal' and q['constraint_violation']<=1e-6
            assert abs(q['simplex_sum']-1)<1e-6;cone_count+=1
            fallback_count+=int(len(q['attempts'])>1)
        max_violation=max(max_violation,r['max_constraint_violation'])
        if family(r['method'])=='mi-SVM':
            y=np.asarray(r['final_labels']);nplus=c['n_bags']*c['bag_size']
            assert len(y)==2*nplus and np.all(y[nplus:]==-1)
            assert np.all(np.sum(y[:nplus].reshape(-1,c['bag_size'])>0,axis=1)>=3)
        if family(r['method'])=='MIL-CPB':
            dm=r['model_diagnostics'];Y=np.asarray(dm['label_vectors']);d=np.asarray(dm['kernel_weights']);nplus=c['n_bags']*c['bag_size']
            assert np.all(Y[nplus:]==-1) and abs(d.sum()-1)<1e-6 and np.min(d)>=0
            assert all(np.all((Y[:nplus,j].reshape(-1,c['bag_size'])>0).sum(axis=1)>=3) for j in range(Y.shape[1]))
            assert dm['bias_convention'].startswith('regularized intercept')
        if r['stop_reason'] in ['max_iter','label_cycle']:limits.append({k:r[k] for k in ['record_id','method','domain','stop_reason']})
    bags=list((root/'bags').glob('*.json'))
    for path in bags:
        b=json.loads(path.read_text());domain=domains[b['domain']];pos=b['positive_video_ids'];neg=b['negative_video_ids']
        ids=[v for bag in pos+neg for v in bag]
        assert len(ids)==len(set(ids))==2*c['n_bags']*c['bag_size']
        assert set(ids)<=set(domain['split_ids']['train'])
        assert all(len(bag)==c['bag_size'] for bag in pos+neg)
        assert all(labels[v]!=b['class_id'] for bag in neg for v in bag)
        assert [sum(labels[v]==b['class_id'] for v in bag) for bag in pos]==[c['bag_size']-round(b['noise']*c['bag_size'])]*c['n_bags']
        assert b['training_instance_truth_used_by_model'] is False
    report=dict(status='passed' if not any(r['status']!='ok' for r in validation) else 'passed_with_failed_validation_candidates',
        validation_only=args.validation_only,validation_fits=len(validation),test_fits=len(tests),APs_recomputed=ap_count,
        validation_failures=sum(r['status']!='ok' for r in validation),conic_subproblems=cone_count,conic_fallbacks=fallback_count,
        source_and_input_hashes='matched',selection_recomputed=True,bag_manifests_checked=len(bags),
        max_constraint_violation=max_violation,outer_limit_or_cycle_fits=len(limits))
    write=lambda name,obj:(root/name).write_text(json.dumps(obj,indent=2)+'\n')
    write('validation_audit.json' if args.validation_only else 'delivery_audit.json',report)
    write('outer_limits.json',limits)
    if not args.validation_only:
        summary=json.loads((root/'summary.json').read_text())
        for s in summary:
            maps=[]
            for seed in c['seeds']:
                sub=[r for r in tests if r['domain']==s['domain'] and r['noise']==s['noise'] and r['method']==s['method'] and r['seed']==seed]
                assert len(sub)==len(domains[s['domain']]['classes'])
                maps.append(float(np.mean([r['average_precision'] for r in sub])))
            np.testing.assert_allclose(maps,s['seed_mAPs'],atol=1e-14)
            assert abs(np.mean(maps)-s['mean_mAP'])<1e-12 and abs(np.std(maps,ddof=1)-s['std_mAP'])<1e-12
        index={(r['domain'],r['noise'],r['method']):r for r in summary}
        comparisons=[]
        for f in c['families']:
            comparisons.extend([(f+' PI over MIL',f+'-PI',f),(f+' DA over PI',f+'-PI-DA',f+'-PI'),
                (f+' joint over KMM',f+'-PI-DA','KMM+'+f+'-PI'),(f+' partial over full',f+'-PI',f+'-PI-full'),
                (f+' aligned over shuffled',f+'-PI',f+'-PI-shuffled'),(f+' MIL-PI over pSVM+',f+'-PI','pSVM+')])
        trends=[]
        for d in c['domains']:
            for noise in c['noise_levels']:
                for name,a,b in comparisons:
                    delta=100*(np.array(index[d,noise,a]['seed_mAPs'])-np.array(index[d,noise,b]['seed_mAPs']))
                    trends.append(dict(domain=d,noise=noise,comparison=name,delta_pp=float(delta.mean()),seed_delta_pp=delta.tolist(),
                        positive_seeds=int(np.count_nonzero(delta>1e-6)),negative_seeds=int(np.count_nonzero(delta< -1e-6))))
        write('trend_checks.json',trends)
        with (root/'metrics.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=['domain','noise','method','mean_mAP','std_mAP']);writer.writeheader();writer.writerows({k:s[k] for k in writer.fieldnames} for s in summary)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
