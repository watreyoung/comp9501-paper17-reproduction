import unittest
from experiments.literature_run import select_scale


class LiteratureSelectionTest(unittest.TestCase):
    def test_macro_validation_only_and_failed_candidate_excluded(self):
        rows=[dict(domain='full',method='RT',split='validate',class_id=c,scale=s,status='ok',average_precision=ap) for s,ap in [(.1,.4),(1.,.7),(10.,.5)] for c in [0,1]]
        rows.append(dict(domain='full',method='RT',split='test',class_id=0,scale=10.,status='ok',average_precision=1.))
        self.assertEqual(select_scale(rows,'full','RT',[0,1],[.1,1.,10.]),1.)
        rows[2]['status']='incomplete'
        self.assertEqual(select_scale(rows,'full','RT',[0,1],[.1,1.,10.]),10.)
        with self.assertRaises(RuntimeError):select_scale(rows,'full','RT',[0,1,2],[.1,1.,10.])


if __name__=='__main__':unittest.main()
