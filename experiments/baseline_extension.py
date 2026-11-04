"""Independent frozen extension runner for representative literature baselines."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import importlib.metadata
import hashlib,json,shutil,time,traceback
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
from .core_run import source_captions
from .paper_alignment import domain_cache
from .data import make_task
from .models import prepare_problem
from .core_models import fit,parameters
from .feature_adaptation import fit_sa,fit_tca
from .multiview_baselines import fit_kcca,fit_svm2k
from .run import sha256_file,write_json,store_record
from mil_pi.kernels import l2_normalize_columns,get_kernel,KernelSpec


def adapted_problem(problem,source,target):
    # Fixed Gaussian bandwidth from adapted training features only.
    K,spec=get_kernel(source.T,spec=KernelSpec('gaussian'))
    cross=get_kernel(target.T,source.T,spec)
    return dict(problem,K=K,test_kernel=cross,Kst=cross.T),spec.gamma


def transforms(data,p,config):
    X=l2_normalize_columns(np.concatenate(data.positive_visual_bags+data.negative_visual_bags,axis=1)).T
    target=l2_normalize_columns(data.test_features).T;n=len(X);d=config['dimension']
    sa=fit_sa(X,target,d)
    sap,gamma=adapted_problem(p,sa.transform_source(X),sa.transform_target(target))
    yield 'SA',sap,dict(dimension=d,kernel_gamma=gamma),dict(source_basis=sa.source_basis,target_basis=sa.target_basis,alignment=sa.alignment)
    Z=np.r_[X,target];K,_=get_kernel(Z.T,spec=KernelSpec('gaussian',p['kernel_gamma']))
    tca=fit_tca(K,n,d,config['tca_mu']);z=tca.transform_kernel(K)
    tcap,gamma=adapted_problem(p,z[:n],z[n:])
    yield 'TCA',tcap,dict(dimension=d,mu=config['tca_mu'],kernel_gamma=gamma,constraint_error=tca.constraint_error,stationarity_error=tca.stationarity_error),dict(coefficients=tca.coefficients,eigenvalues=tca.eigenvalues)
    cca=fit_kcca(p['K'],p['full_tK'],d,config['kcca_regularization'])
    ccap,gamma=adapted_problem(p,cca.transform_visual(p['K']),cca.transform_visual(p['test_kernel']))
    yield 'KCCA',ccap,dict(dimension=d,regularization=config['kcca_regularization'],kernel_gamma=gamma,canonical_correlations=cca.canonical_correlations.tolist()),dict(visual_coef=cca.visual_coef,text_coef=cca.text_coef,visual_column_mean=cca.visual_column_mean,visual_grand_mean=cca.visual_grand_mean)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/baselines.json'))
    parser.add_argument('--output',type=Path,default=Path('results/baselines'))
    parser.add_argument('--phase',choices=['tune','evaluate','all'],default='all')
    args=parser.parse_args();config=json.loads(args.config.read_text());root=args.output;root.mkdir(parents=True,exist_ok=True)
    core=Path(config['core_results']);cm=json.loads((core/'run_manifest.json').read_text());cc=cm['config']
    audit=json.loads((core/'delivery_audit.json').read_text());assert audit['test_fits']==9600 and not audit['validation_only']
    for filename,digest in cm['source_hashes'].items():assert sha256_file(filename)==digest,filename
    files=[p for directory in ['mil_pi','experiments'] for p in sorted(Path(directory).glob('*.py'))]
    identity=dict(config=config,config_sha256=sha256_file(args.config),core_manifest_sha256=sha256_file(core/'run_manifest.json'),
        core_audit_sha256=sha256_file(core/'delivery_audit.json'),cache_sha256=sha256_file(cc['cache']),
        protocol_sha256=sha256_file(config['protocol_file']),source_hashes={str(p):sha256_file(p) for p in files},
        versions={x:importlib.metadata.version(x) for x in ['numpy','scipy','scikit-learn','osqp','cvxpy','clarabel','scs']})
    manifest=root/'run_manifest.json'
    if manifest.exists():assert json.loads(manifest.read_text())==identity,'Run identity changed: use a new directory'
    else:
        write_json(manifest,identity);shutil.copy2(args.config,root/'config_snapshot.json');shutil.copy2(config['protocol_file'],root/'protocol_snapshot.md')
        for file in files:
            dest=root/'source_snapshot'/file;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,dest)
    for directory in ['predictions','bags','transforms']:(root/directory).mkdir(exist_ok=True)
    with np.load(cc['cache']) as f:cache={k:f[k] for k in f.files}
    captions,_=source_captions();lock=Lock();log=root/'records.jsonl'
    rows=[json.loads(line) for line in log.read_text().splitlines()] if log.exists() else [];existing={r['record_id']:r for r in rows}
    selected_path=root/'selected_parameters.json';selected=json.loads(selected_path.read_text()) if selected_path.exists() else {}
    def task(domain,c,cls,seed,noise,split,scales=None):
        tid=hashlib.sha256(json.dumps([domain,cls,seed,noise,split]).encode()).hexdigest()[:20]
        data,bag=make_task(c,cls,seed=seed,noise=noise,n_bags=cc['n_bags'],bag_size=cc['bag_size'],evaluation_split=split)
        p=prepare_problem(data);text=np.concatenate(data.positive_text_bags+data.negative_text_bags,axis=1);p['full_tK']=text.T@text
        write_json(root/'bags'/f'{tid}.json',dict(domain=domain,task_id=tid,**bag))
        # Compute each transform once per bag/target task, reusing it for all C1 candidates and heads.
        def run(method,problem,transform_diag,transform_id):
            for scale in scales if scales is not None else [selected[domain][method]]:
                key=dict(domain=domain,method=method,class_id=cls,seed=seed,noise=noise,split=split,scale=scale)
                rid=hashlib.sha256(json.dumps(key,sort_keys=True).encode()).hexdigest()[:20]
                if rid in existing:continue
                row=dict(key,record_id=rid,task_id=tid,transform_id=transform_id,transform_diagnostics=transform_diag);start=time.perf_counter()
                try:
                    params=parameters('sMIL',scale)
                    if method=='SVM-2K':
                        nplus=cc['n_bags']*cc['bag_size'];costs=np.r_[np.full(nplus,params['C1']),np.ones(nplus)];y=np.r_[np.ones(nplus),-np.ones(nplus)]
                        model=fit_svm2k(p['K'],p['full_tK'],y,costs,costs,coupling=config['svm2k_coupling'],epsilon=config['svm2k_epsilon'])
                        scores=model.predict_visual(p['test_kernel']);diag=dict(objective=model.objective,constraint_violation=model.constraint_violation,agreement_loss=model.agreement_loss)
                        np.savez_compressed(root/'transforms'/f'{rid}_classifier.npz',visual_coef=model.visual_coef,visual_bias=model.visual_bias,text_coef=model.text_coef,text_bias=model.text_bias)
                    else:
                        head='sMIL-PI' if method.endswith('+sMIL-PI') else 'SVM';scores,diag=fit(head,problem,params,seed=seed)
                    if not np.isfinite(scores).all():raise ValueError('Nonfinite predictions')
                    with (root/'predictions'/f'{rid}.tmp').open('wb') as f:np.savez_compressed(f,scores=scores,labels=data.test_labels,video_ids=bag['evaluation_video_ids'])
                    (root/'predictions'/f'{rid}.tmp').replace(root/'predictions'/f'{rid}.npz')
                    row.update(status='ok',average_precision=float(average_precision_score(data.test_labels>0,scores)),parameters=params,diagnostics=diag)
                except Exception as error:row.update(status='failed',error=str(error),traceback=traceback.format_exc())
                row['seconds']=time.perf_counter()-start
                with lock:
                    store_record(log,rows,row);existing[rid]=row
                    print(len(rows),domain,method,cls,seed,noise,split,row['status'],flush=True)
        # A failed transform aborts the task and is retained in task_failures.jsonl.
        for name,problem,diag,arrays in transforms(data,p,config):
            transform_id=tid+'_'+name
            np.savez_compressed(root/'transforms'/f'{transform_id}.npz',**arrays)
            methods=[name+'+SVM',name+'+sMIL-PI'] if name in ['SA','TCA'] else ['KCCA+SVM']
            for method in methods:run(method,problem,diag,transform_id)
        run('SVM-2K',p,dict(coupling=config['svm2k_coupling'],epsilon=config['svm2k_epsilon']),None)
    def guarded_task(*args):
        try:return task(*args)
        except Exception as error:
            with lock:
                with (root/'task_failures.jsonl').open('a') as f:
                    f.write(json.dumps(dict(domain=args[0],class_id=args[2],seed=args[3],noise=args[4],split=args[5],error=str(error),traceback=traceback.format_exc()))+'\n')
            raise
    with threadpool_limits(limits=1),ThreadPoolExecutor(max_workers=config['workers']) as pool:
        for domain in cc['domains']:
            c,dm=domain_cache(cache,domain,n_bags=cc['n_bags'],bag_size=cc['bag_size'],captions=captions)
            assert dm==json.loads((core/f'{domain}_domain.json').read_text());write_json(root/f'{domain}_domain.json',dm)
            if args.phase in ['tune','all']:
                list(pool.map(lambda cls:guarded_task(domain,c,cls,cc['validation_seed'],cc['validation_noise'],'validate',cc['scales']),dm['classes']))
                selected[domain]={}
                for method in config['methods']:
                    options=[]
                    for scale in cc['scales']:
                        sub=sorted([r for r in rows if r['domain']==domain and r['method']==method and r['split']=='validate' and r['scale']==scale],key=lambda r:r['class_id'])
                        if len(sub)==len(dm['classes']) and all(r['status']=='ok' for r in sub):options.append((scale,float(np.mean([r['average_precision'] for r in sub]))))
                    if not options:raise RuntimeError('No complete validation candidate: '+method)
                    selected[domain][method]=max(options,key=lambda x:x[1])[0]
                write_json(selected_path,selected)
            if args.phase in ['evaluate','all']:
                tasks=[(cls,seed,noise) for noise in cc['noise_levels'] for seed in cc['seeds'] for cls in dm['classes']]
                list(pool.map(lambda t:guarded_task(domain,c,*t,'test'),tasks))
    print('Saved',len(rows),'fits',flush=True)


if __name__=='__main__':main()
