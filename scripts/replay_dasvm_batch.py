"""Reconstruct DASVM final SVM from frozen state, preserve original artifacts."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from experiments.core_run import source_captions
from experiments.paper_alignment import domain_cache
from experiments.data import make_task
from experiments.models import prepare_problem
from experiments.run import sha256_file
from mil_pi.kernels import l2_normalize_columns,get_kernel,KernelSpec


def main(root):
    root=Path(root);manifest=json.loads((root/'run_manifest.json').read_text());cc=json.loads((Path(manifest['config']['core_results'])/'run_manifest.json').read_text())['config']
    for f,h in manifest['source_hashes'].items():assert sha256_file(f)==h
    rows=[json.loads(s) for s in (root/'records.jsonl').read_text().splitlines()];rows=[r for r in rows if r['method']=='DASVM' and r['status']!='failed']
    with np.load(cc['cache']) as f:cache={k:f[k] for k in f.files}
    captions,_=source_captions();domains={};out=root/'reconstructed_dasvm_models';out.mkdir(exist_ok=True)
    report_path=out/'replay_records.json';reports=json.loads(report_path.read_text()) if report_path.exists() else {}
    with threadpool_limits(limits=1):
        for row in rows:
            rid=row['record_id']
            if rid in reports:
                assert sha256_file(out/f'{rid}.npz')==reports[rid]['reconstructed_sha256'];continue
            domain=row['domain']
            if domain not in domains:domains[domain]=domain_cache(cache,domain,captions=captions)[0]
            data,bag=make_task(domains[domain],row['class_id'],seed=row['seed'],noise=row['noise'],n_bags=cc['n_bags'],bag_size=cc['bag_size'],evaluation_split=row['split'])
            assert bag==json.loads((root/'bags'/f"{row['task_id']}.json").read_text())
            p=prepare_problem(data);X=l2_normalize_columns(np.concatenate(data.positive_visual_bags+data.negative_visual_bags,axis=1)).T;target=l2_normalize_columns(data.test_features).T
            K,_=get_kernel(np.r_[X,target].T,spec=KernelSpec('gaussian',p['kernel_gamma']))
            h=row['diagnostics']['history'][-1];indices=np.array(h['training_indices'],int)
            source_y=np.r_[np.ones(sum(b.shape[1] for b in data.positive_visual_bags)),-np.ones(sum(b.shape[1] for b in data.negative_visual_bags))]
            labels=np.r_[source_y[h['source_remaining']],h['active_target_labels']];cost=np.array(h['training_costs']);coef=np.zeros(len(K))
            assert len(indices)==len(labels)==len(cost)
            if len(np.unique(labels))==1:bias=float(labels[0]);constant=True
            else:
                clf=SVC(C=1.,kernel='precomputed',tol=1e-7,max_iter=200000).fit(K[np.ix_(indices,indices)],labels,sample_weight=cost)
                assert clf.fit_status_==0;coef[indices[clf.support_]]=clf.dual_coef_[0];bias=float(clf.intercept_[0]);constant=False
            scores=K[len(X):]@coef+bias
            with np.load(root/'predictions'/f'{rid}.npz') as pred:
                error=float(np.max(np.abs(scores-pred['scores'])));np.testing.assert_allclose(scores,pred['scores'],rtol=1e-7,atol=1e-7)
            np.savez_compressed(out/f'{rid}.npz',joint_coef=coef,bias=np.array(bias),kernel_gamma=np.array(p['kernel_gamma']),training_indices=indices)
            reports[rid]=dict(original_prediction_sha256=row['prediction_sha256'],original_model_sha256=row['model_sha256'],reconstructed_sha256=sha256_file(out/f'{rid}.npz'),max_prediction_error=error,constant_solution=constant)
            report_path.write_text(json.dumps(reports,indent=2)+'\n')
    summary=dict(replayed_fits=len(reports),snapshot_dasvm_fits=len(rows),max_prediction_error=max((r['max_prediction_error'] for r in reports.values()),default=0.),script_sha256=sha256_file(__file__),batch_complete=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');main(p.parse_args().root)
