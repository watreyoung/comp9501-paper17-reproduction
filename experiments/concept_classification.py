"""Visual+concept augmentation with the same event bags/costs as core SVM.

Representation training is separate. This module never fits a representation
on event evaluation samples. Concept block weight is explicit and fixed before
evaluation; final per-video normalization matches the core visual pipeline.
"""
import numpy as np
from mil_pi.kernels import l2_normalize_columns,get_kernel,KernelSpec
from .core_models import fit,parameters


def event_rows(video_ids,bag_manifest):
    ids=np.asarray(video_ids)
    if len(set(ids.tolist()))!=len(ids):raise ValueError('Video IDs must be unique')
    lookup={str(v):i for i,v in enumerate(ids)}
    try:
        pos=[np.array([lookup[str(v)] for v in b],int) for b in bag_manifest['positive_video_ids']]
        neg=[np.array([lookup[str(v)] for v in b],int) for b in bag_manifest['negative_video_ids']]
        evaluation=np.array([lookup[str(v)] for v in bag_manifest['evaluation_video_ids']],int)
    except KeyError as e:raise ValueError('Bag ID absent from representation cache') from e
    source=np.concatenate(pos+neg)
    if len(np.unique(source))!=len(source) or np.intersect1d(source,evaluation).size:raise ValueError('Repeated training IDs or source/evaluation overlap')
    if not pos or len(pos)!=len(neg) or len({len(b) for b in pos+neg})!=1:raise ValueError('Core event protocol requires equal positive/negative bag counts and sizes')
    return source,evaluation,len(pos),len(pos[0])


def concept_event_problem(visual,concept_scores,video_ids,bag_manifest,*,concept_weight=1.):
    X=np.asarray(visual,float);Z=np.asarray(concept_scores,float)
    if X.ndim!=2 or Z.ndim!=2 or len(X)!=len(Z) or len(X)!=len(video_ids):raise ValueError('Aligned visual/concept/ID rows required')
    if not np.isfinite(concept_weight) or concept_weight<0:raise ValueError('Finite nonnegative concept weight required')
    source,target,b,m=event_rows(video_ids,bag_manifest)
    # Only accessed rows must be finite; unused/excluded rows may be absent.
    def augmented(rows):
        visual=l2_normalize_columns(X[rows].T).T
        return l2_normalize_columns(np.column_stack([visual,concept_weight*Z[rows]]).T).T
    train=augmented(source);evaluation=augmented(target)
    K,spec=get_kernel(train.T,spec=KernelSpec('gaussian'))
    cross=get_kernel(evaluation.T,train.T,spec)
    return dict(K=K,full_tK=np.zeros_like(K),test_kernel=cross,Kst=cross.T,n_bags=b,bag_size=m,kernel_gamma=spec.gamma)


def classify_concepts(visual,concept_scores,video_ids,bag_manifest,*,scale=1.,concept_weight=1.):
    problem=concept_event_problem(visual,concept_scores,video_ids,bag_manifest,concept_weight=concept_weight)
    scores,diag=fit('SVM',problem,parameters('SVM',scale),seed=bag_manifest['seed'])
    diag.update(concept_weight=concept_weight,concept_dimensions=np.asarray(concept_scores).shape[1],augmentation='unit visual + weighted raw concept decisions, then per-video L2; source-only median RBF')
    return scores,diag
