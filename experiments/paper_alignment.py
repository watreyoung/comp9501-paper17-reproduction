"""Public-data tests of the comparisons in paper Tables 1, 2 and 5.

Both original Full split and a train-only visual-cluster domain stress test.
Classes/target domain chosen using train/validation only, never test labels.
"""
import hashlib
import json
from pathlib import Path
import time
import argparse
from collections import Counter,defaultdict
from zipfile import ZipFile
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
from .data import make_task
from .models import prepare_problem
from .alignment_models import METHODS,aligned_predict
from .run import sha256_file,write_json


def domain_cache(cache,domain,*,n_bags=5,bag_size=5,captions=None):
    c={k:v.copy() for k,v in cache.items()}
    manifest={'domain':domain,'definition':'Published Full split, no constructed domain shift'}
    if domain=='visual_cluster':
        train=cache['splits']=='train'
        normalized=cache['visual']/np.maximum(np.linalg.norm(cache['visual'],axis=1,keepdims=True),1e-12)
        km=KMeans(n_clusters=2,random_state=17,n_init=10).fit(normalized[train])
        assigned=km.predict(normalized)
        source=int(np.argmax(np.bincount(km.labels_,minlength=2)));target=1-source
        keep=(train&(assigned==source))|(~train&(assigned==target))
        c['splits'][~keep]='excluded'
        manifest=dict(domain=domain,definition='Training-fitted two-cluster visual covariate shift; NOT natural cross-dataset transfer',
            kmeans_random_state=17,kmeans_n_init=10,source_cluster=source,target_cluster=target,
            centers_sha256=hashlib.sha256(km.cluster_centers_.tobytes()).hexdigest())
    if domain=='visual_cluster':
        if captions is None:raise ValueError('Source-only caption refit required for visual-cluster protocol')
        train_rows=np.flatnonzero(c['splits']=='train')
        vectorizer=CountVectorizer(max_features=2000,stop_words='english')
        tf=vectorizer.fit_transform([captions[str(c['video_ids'][i])] for i in train_rows])
        c['text']=np.zeros((len(c['video_ids']),tf.shape[1]),dtype=np.float32)
        c['text'][train_rows]=tf.toarray()
        manifest.update(text_vocabulary=vectorizer.get_feature_names_out().tolist(),
                        text_vocabulary_fit_ids=c['video_ids'][train_rows].tolist(),
                        text_refit='Raw source-only TF; all other text rows remain zero')
    classes=[i for i in range(20) if np.count_nonzero((c['splits']=='train')&(c['labels']==i))>=n_bags*bag_size
             and np.count_nonzero((c['splits']=='validate')&(c['labels']==i))>=3]
    manifest.update(classes=classes,class_selection='source train >=25 videos and target validation >=3 positives; test labels unused',
        split_ids={s:c['video_ids'][c['splits']==s].tolist() for s in ['train','validate','test']},
        source_counts={str(i):int(np.count_nonzero((c['splits']=='train')&(c['labels']==i))) for i in classes},
        validation_counts={str(i):int(np.count_nonzero((c['splits']=='validate')&(c['labels']==i))) for i in classes})
    if not classes:raise ValueError('No eligible source classes')
    return c,manifest


def candidate(scale):
    return dict(C=100.*scale,gamma=100.,C2=10.,**{'lambda':100.},sigma=.6)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('results/paper_alignment'))
    args=parser.parse_args(argv);root=args.output;root.mkdir(parents=True,exist_ok=True)
    config=dict(cache='data/prepared/msrvtt.npz',domains=['full','visual_cluster'],n_bags=5,bag_size=5,
                noise_levels=[0.,.2,.4],seeds=[0,1,2],validation_seed=0,validation_noise=.2,
                scales=[.1,1.,10.],methods=METHODS,primary_metric='one-vs-rest mAP',
                family='sMIL-focused paper-question alignment; not full baseline or action replication',
                parameters='C1=100*scale, gamma=100, DA C2=10 and lambda=100 (paper numerical starts)',
                negative_loss='per-instance negative cost 1 in every partial-PI/no-PI classifier',
                full_PI_and_KMM_and_shuffled_parameters='reuse validation-selected sMIL-PI parameters',
                target_labels='validation for selection; test only after model selection; DA/KMM use unlabelled evaluation visuals')
    identity=dict(config=config,cache_sha256=sha256_file(config['cache']),
                  source_hashes={str(p):sha256_file(p) for directory in ['mil_pi','experiments'] for p in sorted(Path(directory).glob('*.py'))})
    manifest_path=root/'run_manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())!=identity:raise ValueError('Changed run identity; use a new directory')
    else:write_json(manifest_path,identity)
    with np.load(config['cache']) as f:cache={k:f[k] for k in f.files}
    archive=Path('data/downloads/msrvtt.zip')
    if sha256_file(archive)!='bb8cc603f9cfd0543dfa03bc88a4dae887f97ffcd21a59fc97aecc2296311828':raise ValueError('Caption archive changed')
    with ZipFile(archive) as z:annotation=json.loads(z.read('msrvtt/Annotations/MSRVTT_data.json'))
    caption_lists=defaultdict(list)
    for row in annotation['sentences']:caption_lists[row['video_id']].append(row['caption'])
    captions={video:' '.join(values) for video,values in caption_lists.items()}
    log=root/'records.jsonl';rows=[json.loads(s) for s in log.read_text().splitlines()] if log.exists() else []
    existing={r['record_id']:r for r in rows};problems={}
    def run(domain,c,method,cls,seed,noise,split,p,scale):
        key=dict(domain=domain,method=method,class_id=cls,seed=seed,noise=noise,split=split,parameters=p,candidate=scale)
        rid=hashlib.sha256(json.dumps(key,sort_keys=True).encode()).hexdigest()[:20]
        if rid in existing:return existing[rid]
        taskkey=(domain,cls,seed,noise,split)
        if taskkey not in problems:
            data,bag=make_task(c,cls,seed=seed,noise=noise,n_bags=5,bag_size=5,evaluation_split=split)
            problem=prepare_problem(data)
            text=np.concatenate(data.positive_text_bags+data.negative_text_bags,axis=1)
            problem['full_tK']=text.T@text
            # Bound memory: each problem includes all target-by-source kernels.
            if len(problems)>25:problems.clear()
            problems[taskkey]=(data,bag,problem)
        data,bag,problem=problems[taskkey];start=time.perf_counter()
        scores,diagnostics=aligned_predict(method,problem,p,seed=seed)
        ap=float(average_precision_score(data.test_labels>0,scores))
        record=dict(**key,record_id=rid,status='ok',average_precision=ap,seconds=time.perf_counter()-start,
                    kernel_gamma=problem['kernel_gamma'],**diagnostics)
        pred=root/'predictions';pred.mkdir(exist_ok=True)
        np.savez_compressed(pred/f'{rid}.npz',scores=scores,labels=data.test_labels,video_ids=bag['evaluation_video_ids'])
        bags=root/'bags';bags.mkdir(exist_ok=True)
        task_id=hashlib.sha256(json.dumps(taskkey).encode()).hexdigest()[:20]
        if not (bags/f'{task_id}.json').exists():write_json(bags/f'{task_id}.json',dict(domain=domain,**bag))
        with log.open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
        rows.append(record);existing[rid]=record
        print(f'{domain} {split} {method} class={cls} seed={seed} noise={noise} C={p["C"]}: {ap:.5f}',flush=True)
        return record
    with threadpool_limits(limits=1):
        selected={};validation={}
        for domain in config['domains']:
            c,dm=domain_cache(cache,domain,captions=captions);write_json(root/f'{domain}_domain.json',dm)
            classes=dm['classes'];selected[domain]={};validation[domain]={}
            for method in ['SVM','sMIL','pSVM+','sMIL-PI','sMIL-PI-DA']:
                options=[]
                for scale in config['scales']:
                    p=candidate(scale)
                    fits=[run(domain,c,method,cls,0,.2,'validate',p,scale) for cls in classes]
                    options.append(dict(scale=scale,mAP=float(np.mean([r['average_precision'] for r in fits]))))
                best=max(options,key=lambda r:r['mAP']);selected[domain][method]=candidate(best['scale'])
                validation[domain][method]=options
            for method in ['sMIL-PI-full','KMM+sMIL-PI','sMIL-PI-shuffled']:
                selected[domain][method]=selected[domain]['sMIL-PI'].copy()
            write_json(root/'selected_parameters.json',selected);write_json(root/'validation_candidates.json',validation)
            for noise in config['noise_levels']:
                for seed in config['seeds']:
                    for cls in classes:
                        for method in METHODS:run(domain,c,method,cls,seed,noise,'test',selected[domain][method],'selected_on_validation')
    summary=[]
    for domain in config['domains']:
        for noise in config['noise_levels']:
            for method in METHODS:
                maps=[float(np.mean([r['average_precision'] for r in rows if r['domain']==domain and r['noise']==noise and r['method']==method and r['split']=='test' and r['seed']==seed])) for seed in config['seeds']]
                summary.append(dict(domain=domain,noise=noise,method=method,mean_mAP=float(np.mean(maps)),std_mAP=float(np.std(maps,ddof=1)),seed_mAPs=maps))
    write_json(root/'summary.json',summary)
    print('Saved',root/'summary.json',len(rows),'fits',flush=True)

if __name__=='__main__':main()
