# Data provenance and limits

The final public protocol uses a fixed MSR-VTT feature mirror with ImageBind visual features and captions. The prepared cache contains 10,000 videos, source-training vocabulary, video IDs, labels, and explicit train/validation/test splits.

The original paper's Flickr/YouTube source inputs and Kodak/CCV/HMDB51 evaluation inputs are not available in the required four-MAT form. The visual-cluster setting is a constructed shift inside the public data; it is not a natural cross-dataset adaptation benchmark.

The source revision and checksums are recorded in `data/prepared/`; the bibliographic entries are included in the repository README. The binary cache is intentionally excluded from GitHub and must be restored separately before a full run.
