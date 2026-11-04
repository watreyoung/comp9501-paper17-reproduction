import unittest
from experiments.full_adaptation import select_validation_family


class FamilySelectionTest(unittest.TestCase):
    def rows(self):
        return [dict(split='validate',head=h,scale=s,class_id=c,status='ok',average_precision=ap)
                for h,ap in [('sMIL-PI',.2),('mi-SVM-PI',.5),('MIL-CPB-PI',.4)] for s in [.1,1.] for c in [0,1]]

    def test_complete_macro_and_deterministic_ties(self):
        choice=select_validation_family(self.rows(),[0,1],[.1,1.])
        self.assertEqual(choice,dict(head='mi-SVM-PI',scale=.1,validation_map=.5))

    def test_failure_excluded_and_no_test_selection(self):
        rows=self.rows()
        for r in rows:
            if r['head']=='mi-SVM-PI':r['status']='failed'
        self.assertEqual(select_validation_family(rows,[0,1],[.1,1.])['head'],'MIL-CPB-PI')
        rows[0]['split']='test'
        with self.assertRaises(ValueError):select_validation_family(rows,[0,1],[.1,1.])

    def test_duplicate_and_incomplete_candidates(self):
        rows=self.rows();rows.append(rows[0].copy())
        with self.assertRaises(ValueError):select_validation_family(rows,[0,1],[.1,1.])
        with self.assertRaises(ValueError):select_validation_family(self.rows(),[0,1,2],[.1,1.])


if __name__=='__main__':unittest.main()
