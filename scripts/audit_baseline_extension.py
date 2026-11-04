"""Independent coverage, input, AP, bag and selection audit for the extension."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=Path('results/baselines'))
    parser.add_argument('--validation-only',action='store_true');args=parser.parse_args();root=args.results
    read=lambda name:json.loads((root/name).read_text())
    m=read('run_manifest.json');c=m['config'];core=Path(c['core_results']);cm=json.loads((core/'run_manifest.json').read_text());cc=cm['config']
    for name,key in [('config_snapshot.json','config_sha256'),('protocol_snapshot.md','protocol_sha256')]:assert digest(root/name)==m[key]
    assert digest(core/'run_manifest.json')==m['core_manifest_sha256']
    assert digest(core/'delivery_audit.json')==m['core_audit_sha256']
    assert digest(cc['cache'])==m['cache_sha256']==cm['cache_sha256']
    for file,h in m['source_hashes'].items():assert digest(file)==digest(root/'source_snapshot'/file)==h,file
    domains={d:read(d+'_domain.json') for d in cc['domains']}
    for d,dm in domains.items():assert dm==json.loads((core/f'{d}_domain.json').read_text())
    with np.load(cc['cache']) as f:labels=dict(zip(f['video_ids'],f['labels']))
    rows=[json.loads(line) for line in (root/'records.jsonl').read_text().splitlines()]
    assert len(rows)==len({r['record_id'] for r in rows})
    validation=[r for r in rows if r['split']=='validate'];tests=[r for r in rows if r['split']=='test']
    vkeys={(d,cls,method,scale) for d,dm in domains.items() for cls in dm['classes'] for method in c['methods'] for scale in cc['scales']}
    assert len(validation)==len(vkeys)
    assert {(r['domain'],r['class_id'],r['method'],r['scale']) for r in validation}==vkeys
    assert all(r['seed']==cc['validation_seed'] and r['noise']==cc['validation_noise'] for r in validation)
    selected=read('selected_parameters.json')
    for d,dm in domains.items():
        for method in c['methods']:
            options=[]
            for scale in cc['scales']:
                sub=sorted([r for r in validation if r['domain']==d and r['method']==method and r['scale']==scale],key=lambda r:r['class_id'])
                if len(sub)==len(dm['classes']) and all(r['status']=='ok' for r in sub):options.append((scale,float(np.mean([r['average_precision'] for r in sub]))))
            assert options and selected[d][method]==max(options,key=lambda x:x[1])[0]
    if not args.validation_only:
        tkeys={(d,cls,method,seed,noise) for d,dm in domains.items() for cls in dm['classes'] for method in c['methods'] for seed in cc['seeds'] for noise in cc['noise_levels']}
        assert len(tests)==len(tkeys)
        assert {(r['domain'],r['class_id'],r['method'],r['seed'],r['noise']) for r in tests}==tkeys
        assert all(r['status']=='ok' for r in tests)
    ap_count=0;max_constraint=0.;checked=set();transform_checked=set()
    for r in rows:
        if r['status']!='ok':assert r['split']=='validate' and r['error'];continue
        if r['split']=='test':assert r['scale']==selected[r['domain']][r['method']]
        assert r['parameters']['C1']==100*r['scale']
        assert r['parameters']['gamma']==100 and r['parameters']['sigma']==cc['sigma']
        with np.load(root/'predictions'/f'{r["record_id"]}.npz') as f:
            assert set(f['video_ids'])==set(domains[r['domain']]['split_ids'][r['split']])
            assert len(f['video_ids'])==len(set(f['video_ids']))
            truth=np.array([1 if labels[v]==r['class_id'] else -1 for v in f['video_ids']])
            np.testing.assert_array_equal(truth,f['labels']);assert np.isfinite(f['scores']).all()
            assert abs(average_precision_score(truth>0,f['scores'])-r['average_precision'])<1e-12
        ap_count+=1;diag=r['diagnostics'];td=r['transform_diagnostics']
        violation=diag.get('constraint_violation',diag.get('max_constraint_violation',0.));assert violation<=1e-6
        max_constraint=max(max_constraint,violation)
        if r['method'].startswith('TCA'):
            assert td['constraint_error']<=1e-6 and td['stationarity_error']<=1e-6
        if r['transform_id'] and r['transform_id'] not in transform_checked:
            with np.load(root/'transforms'/f'{r["transform_id"]}.npz') as f:assert all(np.isfinite(f[k]).all() for k in f.files)
            transform_checked.add(r['transform_id'])
        if r['method']=='SVM-2K':
            with np.load(root/'transforms'/f'{r["record_id"]}_classifier.npz') as f:assert all(np.isfinite(f[k]).all() for k in f.files)
        if r['task_id'] not in checked:
            b=read('bags/'+r['task_id']+'.json');original=json.loads((core/'bags'/f'{r["task_id"]}.json').read_text())
            assert b==original,'Extension must reuse core bags and target IDs'
            checked.add(r['task_id'])
    summary=[]
    if not args.validation_only:
        for d in domains:
            for noise in cc['noise_levels']:
                for method in c['methods']:
                    maps=[float(np.mean([r['average_precision'] for r in tests if (r['domain'],r['noise'],r['seed'],r['method'])==(d,noise,seed,method)])) for seed in cc['seeds']]
                    summary.append(dict(domain=d,noise=noise,method=method,seed_mAPs=maps,mean_mAP=float(np.mean(maps)),std_mAP=float(np.std(maps,ddof=1))))
        (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    report=dict(status='passed' if all(r['status']=='ok' for r in validation) else 'passed_with_failed_validation_candidates',validation_only=args.validation_only,
        validation_fits=len(validation),test_fits=len(tests),APs_recomputed=ap_count,core_bags_matched=len(checked),transforms_checked=len(transform_checked),max_constraint_violation=max_constraint,
        source_and_input_hashes='matched',selection_recomputed=True)
    (root/('validation_audit.json' if args.validation_only else 'delivery_audit.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
