import unittest
import numpy as np
from experiments.concept_data import SourceMinMax,miml_source_bags,classeme_word_bags


class ConceptDataTest(unittest.TestCase):
    def test_training_scaler_unseen_extremes_and_constants(self):
        model=SourceMinMax.fit([[1,3],[5,3]])
        np.testing.assert_allclose(model.transform([[9,100],[-3,-9]]),[[2,.5],[-1,.5]])

    def test_miml_partition_union_and_withheld_caption_independence(self):
        text=np.array([[1,0],[0,1],[0,0],[1,1],[np.nan,np.nan]])
        split=np.array(['train']*4+['test'])
        bags,y=miml_source_bags(text,split,bag_size=3)
        np.testing.assert_array_equal(np.sort(np.concatenate(bags)),np.arange(4))
        self.assertEqual([len(b) for b in bags],[3,1])
        for b,label in zip(bags,y):np.testing.assert_array_equal(label,np.any(text[b]>0,axis=0))
        text[4]=[1e9,1e9]
        b2,y2=miml_source_bags(text,split,bag_size=3)
        np.testing.assert_array_equal(y,y2)
        for a,b in zip(bags,b2):np.testing.assert_array_equal(a,b)

    def test_rare_word_retained_without_repeats_or_target_rows(self):
        text=np.zeros((34,2));text[:3,0]=1;text[:28,1]=1;text[-1]=np.nan
        split=['train']*33+['validate']
        pos,neg=classeme_word_bags(text,split,0)
        self.assertEqual([len(b) for b in pos],[3])
        self.assertEqual(len(neg),5)
        used=np.concatenate(pos+neg)
        self.assertEqual(len(used),len(np.unique(used)))
        self.assertTrue(np.all(text[np.concatenate(pos),0]>0))
        self.assertTrue(np.all(text[np.concatenate(neg),0]==0))
        self.assertLess(used.max(),33)


if __name__=='__main__':unittest.main()
