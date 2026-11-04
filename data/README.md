# Data and generated files

This GitHub snapshot keeps provenance metadata and task manifests, but intentionally omits the feature cache, downloaded archive, and MAT binaries. Those files are large derived artifacts and should not be treated as source code.

The recorded public protocol uses:

- `data/prepared/msrvtt.npz`: prepared visual/text features, labels, video IDs, and train/validation/test split.
- `data/prepared/msrvtt_manifest.json`: source revision, dimensions, split information, and protocol checks.
- `data/prepared/msrvtt_vocabulary.json`: vocabulary fitted from source-training captions only.
- `data/prepared/visual_feature_sha256.json`: per-source-file checksums.
- `data/tasks/*/task_manifest.json`: manifests for the four-MAT compatibility exports.

Restore the fixed cache from the source recorded in `docs/data_search.md` and verify the included manifest before running `experiments.run`. The download step may require network access and the upstream dataset terms apply. The runner will record the cache hash in `run_manifest.json`; do not reuse an output directory after changing the cache.

The original four-MAT interface is:

```text
visual_bags.mat
text_bags.mat
test_features.mat
test_labels.mat
```

For a dependency-free local smoke test, generate these files with `python examples/make_synthetic_mat_data.py /tmp/mil_pi_synthetic`.
