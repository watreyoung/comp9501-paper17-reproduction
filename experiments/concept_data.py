"""Source-only data construction for public-data word-concept baselines.

This is an explicit replacement-data protocol, not a claim about unavailable
original video bags. Event labels and validation/test captions are not inputs.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class SourceMinMax:
    minimum: np.ndarray
    span: np.ndarray

    @classmethod
    def fit(cls,source_instances):
        X=np.asarray(source_instances,float)
        if X.ndim!=2 or not len(X) or not np.isfinite(X).all():raise ValueError('Finite source matrix required')
        return cls(X.min(axis=0),np.ptp(X,axis=0))

    def transform(self,X):
        X=np.asarray(X,float)
        if X.ndim!=2 or X.shape[1]!=len(self.minimum) or not np.isfinite(X).all():raise ValueError('Finite shared feature dimension required')
        # Do not clip unseen values: preserve the training-fitted affine map.
        return np.divide(X-self.minimum,self.span,out=np.full_like(X,.5),where=self.span>0)


def source_rows(splits):
    rows=np.flatnonzero(np.asarray(splits)=='train')
    if not len(rows):raise ValueError('No source training rows')
    return rows


def miml_source_bags(text,splits,*,bag_size=5,seed=0):
    """Partition each source video once; labels are union of caption words.

    Include the final short bag. Word-label union is a documented public-data
    construction; the target paper does not specify its MIML bag recipe.
    """
    if bag_size<1:raise ValueError('Positive bag size required')
    rows=source_rows(splits);T=np.asarray(text)
    if T.ndim!=2 or len(T)!=len(splits):raise ValueError('Text rows must match split rows')
    # Access only source rows, so withheld captions may be NaN or unavailable.
    if not np.isfinite(T[rows]).all() or np.any(T[rows]<0):raise ValueError('Nonnegative finite source TF required')
    rows=np.random.default_rng(seed).permutation(rows)
    bags=[rows[i:i+bag_size] for i in range(0,len(rows),bag_size)]
    labels=np.array([np.any(T[b]>0,axis=0) for b in bags],dtype=np.uint8)
    return bags,labels


def classeme_word_bags(text,splits,word,*,n_bags=5,bag_size=5,seed=0):
    """Caption-retrieved positive/negative bags, no duplicate video per word.

    Cap each side at n_bags*bag_size. For rare words retain every available
    example in short bags instead of duplicating examples or dropping words.
    No extra event-label contamination is added to noisy caption retrieval.
    """
    if n_bags<1 or bag_size<1:raise ValueError('Positive bag counts and sizes required')
    rows=source_rows(splits);T=np.asarray(text)
    if T.ndim!=2 or len(T)!=len(splits) or not 0<=word<T.shape[1]:raise ValueError('Invalid text matrix or word')
    values=T[rows,word]
    if not np.isfinite(values).all() or np.any(values<0):raise ValueError('Nonnegative finite source TF required')
    # Independent per-word RNG: results do not depend on vocabulary iteration.
    rng=np.random.default_rng(np.random.SeedSequence([seed,int(word)]))
    def make(mask):
        selected=rng.permutation(rows[mask])[:n_bags*bag_size]
        return [selected[i:i+bag_size] for i in range(0,len(selected),bag_size)]
    positive=make(values>0);negative=make(values==0)
    if not positive or not negative:raise ValueError('Word lacks a source positive or negative class; report explicitly')
    return positive,negative
