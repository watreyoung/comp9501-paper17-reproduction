"""Reconstruct predictions and the primal objective from saved SVM-2K models."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,json
import numpy as np
from threadpoolctl import threadpool_limits
from experiments.core_run import source_captions
from experiments.paper_alignment import domain_cache
from experiments.data import make_task
from experiments.models import prepare_problem


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=Path('results/baselines'))
    parser.add_argument('--split',choices=['validate','test','all'],default='all')
    args=parser.parse_args();root=args.results;m=json.loads((root/'run_manifest.json').read_text());config=m['config']
    core=Path(config['core_results']);cc=json.loads((core/'run_manifest.json').read_text())['config']
    rows=[json.loads(line) for line in (root/'records.jsonl').read_text().splitlines()]
    rows=[r for r in rows if r['method']=='SVM-2K' and r['status']=='ok' and (args.split=='all' or r['split']==args.split)]
    assert rows,'No completed models to inspect'
    with np.load(cc['cache']) as f:cache={k:f[k] for k in f.files}
    captions,_=source_captions();max_score=0.;max_objective=0.;count=0;tasks=0
    with threadpool_limits(limits=1):
        for domain in cc['domains']:
            c,_=domain_cache(cache,domain,n_bags=cc['n_bags'],bag_size=cc['bag_size'],captions=captions)
            subset=[r for r in rows if r['domain']==domain]
            keys=sorted({(r['class_id'],r['seed'],r['noise'],r['split']) for r in subset})
            for cls,seed,noise,split in keys:
                data,bag=make_task(c,cls,seed=seed,noise=noise,n_bags=cc['n_bags'],bag_size=cc['bag_size'],evaluation_split=split)
                problem=prepare_problem(data);text=np.concatenate(data.positive_text_bags+data.negative_text_bags,axis=1);T=text.T@text
                nplus=cc['n_bags']*cc['bag_size'];y=np.r_[np.ones(nplus),-np.ones(nplus)];tasks+=1
                for r in [r for r in subset if (r['class_id'],r['seed'],r['noise'],r['split'])==(cls,seed,noise,split)]:
                    with np.load(root/'transforms'/f'{r["record_id"]}_classifier.npz') as f:
                        av=f['visual_coef'];at=f['text_coef'];bv=float(f['visual_bias']);bt=float(f['text_bias'])
                    scores=problem['test_kernel']@av+bv
                    with np.load(root/'predictions'/f'{r["record_id"]}.npz') as f:
                        np.testing.assert_array_equal(f['video_ids'],bag['evaluation_video_ids'])
                        error=float(np.max(np.abs(scores-f['scores'])))
                    assert error<=1e-8;max_score=max(max_score,error)
                    fv=problem['K']@av+bv;ft=T@at+bt
                    costs=np.r_[np.full(nplus,r['parameters']['C1']),np.ones(nplus)]
                    loss_v=np.maximum(1-y*fv,0);loss_t=np.maximum(1-y*ft,0)
                    agreement=np.maximum(np.abs(fv-ft)-config['svm2k_epsilon'],0).sum()
                    objective=.5*(av@problem['K']@av+at@T@at)+costs@(loss_v+loss_t)+config['svm2k_coupling']*agreement
                    gap=abs(objective-r['diagnostics']['objective'])/max(1.,abs(objective))
                    assert gap<=2e-6,(r['record_id'],gap)
                    assert abs(agreement-r['diagnostics']['agreement_loss'])<=1e-6
                    max_objective=max(max_objective,float(gap));count+=1
    report=dict(status='passed_for_saved_models',split=args.split,models_checked=count,tasks_reconstructed=tasks,max_prediction_error=max_score,max_relative_primal_objective_gap=max_objective,
                scope='Saved successful models only; full task coverage is checked by audit_baseline_extension.py')
    out=root/('saved_svm2k_audit_'+args.split+'.json');out.write_text(json.dumps(report,indent=2)+'\n');print(out.read_text())


if __name__=='__main__':main()
