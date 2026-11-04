import unittest
from experiments.full_adaptation import HEADS
from experiments.full_adaptation_run import selections


class SelectionTests(unittest.TestCase):
    def rows(self):
        return [dict(domain='d', adapter='GFK', head=h, split='validate',
                     scale=s, class_id=c, status='ok', average_precision=.5)
                for h in HEADS for s in [.1, 1.] for c in [0, 1]]

    def test_incomplete_high_scoring_candidate_cannot_win(self):
        rows = self.rows()
        for r in rows:
            if r['head'] == 'mi-SVM-PI' and r['scale'] == 1.:
                r['average_precision'] = .99
                if r['class_id'] == 1:
                    r['status'] = 'incomplete'
        # A perfect test score must not influence validation choice.
        rows.append(dict(domain='d', adapter='GFK', head='MIL-CPB-PI',
                         split='test', scale=1., class_id=0, status='ok', average_precision=1.))
        result = selections(rows, 'd', [0, 1], {'adapters': ['GFK']}, [.1, 1.])['GFK']
        self.assertEqual(result['family']['head'], 'sMIL-PI')
        self.assertEqual(result['family']['scale'], .1)
        self.assertEqual(result['heads']['mi-SVM-PI']['scale'], .1)

    def test_missing_class_blocks_head_delivery(self):
        rows = [r for r in self.rows() if not (r['head'] == 'SVM' and r['class_id'] == 1)]
        with self.assertRaises(RuntimeError):
            selections(rows, 'd', [0, 1], {'adapters': ['GFK']}, [.1, 1.])

    def test_duplicate_identity_rejected(self):
        rows = self.rows()
        rows.append(rows[0].copy())
        with self.assertRaises(ValueError):
            selections(rows, 'd', [0, 1], {'adapters': ['GFK']}, [.1, 1.])
