import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from scipy.io import savemat
from sklearn.metrics import average_precision_score
from mil_pi.kernels import average_precision, as_feature_matrix
from mil_pi.data import load_mat_dataset
from mil_pi.algorithms import KernelModel, mi_svm_pi, mi_svm_pi_da
from mil_pi.qp import solve_qp, QPError
from experiments.run import sha256_file, store_record

class RevisionSafetyTest(unittest.TestCase):
    def test_ap_ties_order_independent_and_matches_runner(self):
        for y in [[1,-1],[-1,1],[1,1,-1,-1]]:
            scores=np.zeros(len(y))
            self.assertEqual(average_precision(y,scores),average_precision_score(np.array(y)>0,scores))
        with self.assertRaises(ValueError):average_precision([1],[np.nan])

    def test_failed_feasible_solver_is_rejected_without_context(self):
        result=SimpleNamespace(x=np.zeros(1),success=False,message='iteration limit',status=9,nit=1)
        with patch('mil_pi.qp.minimize',return_value=result):
            with self.assertRaises(QPError):solve_qp(np.eye(1),np.zeros(1),lb=np.zeros(1))

    def test_feature_cache_hash_changes_with_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'cache';p.write_bytes(b'original');before=sha256_file(p)
            p.write_bytes(b'replaced');self.assertNotEqual(before,sha256_file(p))

    def test_retry_replaces_current_record_and_archives_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            log=Path(tmp)/'records.jsonl';records=[]
            store_record(log,records,dict(record_id='a',status='failed'))
            store_record(log,records,dict(record_id='a',status='ok'))
            self.assertEqual(len(records),1)
            self.assertEqual(json.loads(log.read_text())['status'],'ok')
            self.assertEqual(json.loads(log.with_name('superseded_attempts.jsonl').read_text())['status'],'failed')

    def test_nonfinite_features_rejected(self):
        for value in [np.nan,np.inf,-np.inf]:
            with self.assertRaises(ValueError):as_feature_matrix([[value]])

    def test_mat_rejects_misaligned_bags_and_nonfinite_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            def bags(a):
                b=np.empty(1,dtype=[('features','O')]);b[0]['features']=a;return b
            savemat(p/'visual_bags.mat',dict(pos_bags=bags(np.ones((2,2))),neg_bags=bags(np.ones((2,2)))))
            savemat(p/'test_features.mat',dict(test_features=np.ones((2,2))))
            savemat(p/'test_labels.mat',dict(test_labels=[1,-1]))
            savemat(p/'text_bags.mat',dict(pos_bags=bags(np.ones((3,3))),neg_bags=bags(np.ones((3,2)))))
            with self.assertRaisesRegex(ValueError,'instance counts'):load_mat_dataset(p)
            savemat(p/'text_bags.mat',dict(pos_bags=bags(np.ones((3,2))),neg_bags=bags(np.ones((3,2)))))
            savemat(p/'test_labels.mat',dict(test_labels=[np.nan,-1]))
            with self.assertRaisesRegex(ValueError,'test_labels'):load_mat_dataset(p)

    def test_mi_outer_logs_stability_and_cap_without_changing_training_labels(self):
        for da in [False,True]:
            target='mil_pi.algorithms.train_mi_svm_pi'+('_da' if da else '')
            seen=[]
            def train(labels,*args):
                seen.append(labels.copy())
                return KernelModel(coef=np.array([1.,2.,0.,0.]),support_indices=np.arange(4),sv_coef=np.array([1.,2.,0.,0.]),y=labels.copy(),objective=-len(seen))
            for cap,reason in [(1,'max_iter'),(5,'labels_stable')]:
                seen.clear()
                with patch(target,side_effect=train):
                    args=(np.eye(4),np.eye(4))
                    if da:args+= (np.eye(4),np.ones((4,2)))
                    args+=([2],[2],dict(pos_ratio=1.,max_iter=cap))
                    m=(mi_svm_pi_da if da else mi_svm_pi)(*args)
                self.assertEqual(m.stop_reason,reason)
                self.assertEqual(m.outer_iterations,len(seen))
                self.assertEqual(len(m.objective_history),len(seen))
                self.assertEqual(len(m.label_changes),len(seen))
                np.testing.assert_array_equal(m.y,seen[-1])

    def test_runner_nondefault_bags_retry_and_cache_identity(self):
        from experiments.run import main
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cache=root/'cache.npz';config=root/'config.json';out=root/'result'
            labels=np.tile([0,1],30)
            np.savez(cache,visual=np.ones((60,2)),text=np.ones((60,3)),labels=labels,
                     video_ids=np.array([f'v{i}' for i in range(60)]),
                     splits=np.array(['train']*40+['validate']*10+['test']*10))
            c=dict(cache=str(cache),bag_size=10,n_bags_per_sign=1,sigma=.6,
                   classes=[0],seeds=[0],noise_levels=[0.],methods=['SVM'],
                   regularization_scales=[1.],validation_seed=0,validation_noise=0.,
                   outer_max_iter=2,mkl_max_iter=2,solver_max_iter=100,qp_backend='osqp')
            config.write_text(json.dumps(c))
            args=['--config',str(config),'--output',str(out)]
            with patch('experiments.run.fit_predict',side_effect=RuntimeError('forced failure')):
                with self.assertRaisesRegex(RuntimeError,'No fully successful'):main(args)
            with patch('experiments.run.fit_predict',return_value=(np.zeros(10),{})) as fit:
                main(args+['--retry-failed'])
                self.assertEqual(fit.call_count,2)
                main(args)
                self.assertEqual(fit.call_count,2)  # successful records are reused
            self.assertEqual(load_mat_dataset(out/'mat_example').bag_size,10)
            records=[json.loads(x) for x in (out/'records.jsonl').read_text().splitlines()]
            self.assertEqual(len(records),2)
            self.assertTrue(all(r['status']=='ok' for r in records))
            with cache.open('ab') as stream:stream.write(b'cache replaced')
            with self.assertRaisesRegex(ValueError,'Code/config/data changed'):main(args)

if __name__=='__main__':unittest.main()
