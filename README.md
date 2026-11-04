# Reproduction of Exploiting Privileged Information from Web Data

This repository is the **COMP9501 group assignment** reproduction for Paper 17.

**Group members:** Kaiyan Zhang, Yumin Guo, Zezhou Yang, and Zhenyi Fan.

This repository contains our **COMP9501 course assignment** for Paper 17:

> Niu, Li and Xu, “Exploiting Privileged Information from Web Data for Action and Event Recognition.”

The code implements the paper’s MIL, privileged-information, and domain-adaptation mechanisms and evaluates them under a reproducible public-data protocol. MIL (multiple-instance learning) is one component of the paper; it is not the paper title.

## What is included

- Core MIL-PI algorithms and QP solvers in [`mil_pi/`](mil_pi/).
- Data preparation, bag construction, PI/DA models, and experiment runners in [`experiments/`](experiments/).
- Stable experiment configurations in [`configs/`](configs/).
- Audits, reports, solver checks, and replay utilities in [`scripts/`](scripts/).
- Regression and numerical-safety tests in [`tests/`](tests/).
- Original MATLAB entry points for comparison in [`reference/matlab/`](reference/matlab/).
- Protocol and implementation notes in [`docs/`](docs/).
- Saved experiment summaries and audit records in [`results/`](results/).

## Start with the results

The repository includes the actual saved outputs from the completed runs used for the reproduction section of our presentation. The main tables are generated from the committed machine-readable summaries; they are not illustrative examples or placeholder values.

| Purpose | Link |
| --- | --- |
| Final tables and conclusions | [`results/final_report.md`](results/final_report.md) |
| Core MIL/PI/DA tables | [`results/core/REPORT.md`](results/core/REPORT.md) · [`results/core/summary.json`](results/core/summary.json) |
| Core audit | [`results/core/delivery_audit.json`](results/core/delivery_audit.json) |
| Representative baseline tables | [`results/baselines/REPORT.md`](results/baselines/REPORT.md) · [`results/baselines/summary.json`](results/baselines/summary.json) |
| Literature-baseline summaries | [`results/literature/REPORT.md`](results/literature/REPORT.md) · [`results/literature/summary.json`](results/literature/summary.json) |
| Combined comparison data | [`results/summary.json`](results/summary.json) |

The saved evidence contains 10,656 audited core fits, 3,456 audited representative-baseline fits, and 2,880 audited literature-baseline fits. Together these are the 16,992 fits reported in the presentation, including 14,880 successful test fits. The core audit recomputes AP values, task bags, source and input hashes, and validation-based parameter selection; twenty failed validation candidates are retained in the audit and excluded from selection. The detailed coverage inventory is in [`results/coverage.md`](results/coverage.md).

The committed results use the same public-data scope as the presentation: MSR-VTT, ImageBind visual features, source-training captions, full and visual-cluster settings, five seeds, 0/20/40% positive-bag contamination, five positive and five negative bags of five instances, and validation-only parameter selection.

## Mapping from the presentation to the repository

| Reproduction topic | Repository entry |
| --- | --- |
| Dataset source and limitations | [`docs/data_search.md`](docs/data_search.md) · [`data/README.md`](data/README.md) |
| Frozen core protocol | [`configs/core.json`](configs/core.json) · [`docs/core_protocol.md`](docs/core_protocol.md) |
| MIL → PI → DA comparisons | [`experiments/core_run.py`](experiments/core_run.py) · [`results/core/`](results/core/) |
| Full, partial, and shuffled PI controls | [`mil_pi/`](mil_pi/) · [`results/core/EVIDENCE.md`](results/core/EVIDENCE.md) |
| SA, TCA, KCCA, and SVM-2K | [`configs/baselines.json`](configs/baselines.json) · [`results/baselines/`](results/baselines/) |
| KMM+SVM, STM, RT, and DASVM | [`configs/literature.json`](configs/literature.json) · [`results/literature/`](results/literature/) |
| Final interpretation and limitations | [`results/final_report.md`](results/final_report.md) |

The table in [`results/coverage.md`](results/coverage.md) separates completed saved evidence from implemented branches that were intentionally excluded from the final comparison. This keeps the repository aligned with the reproduction claims in the presentation without implying a full reproduction of every original-paper experiment.

## Reproduce the checks

The dependency file pins the environment used for the saved summaries (Python 3.12.2):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The test suite uses generated arrays and temporary MAT files, so it does not require the MSR-VTT cache. A small four-MAT fixture can be generated with:

```bash
python examples/make_synthetic_mat_data.py /tmp/mil_pi_synthetic
```

A full public-data rerun requires `data/prepared/msrvtt.npz`, which is a large derived third-party artifact and is intentionally excluded. After restoring and verifying that cache, use new output directories:

```bash
python -m experiments.core_run --config configs/core.json --output results/core_rerun
python -m experiments.baseline_extension --config configs/baselines.json --output results/baselines_rerun
python -m experiments.literature_run --config configs/literature.json --output results/literature_rerun
```

The runners record configuration, source, protocol, and cache identities and refuse to mix incompatible outputs.

## Scope and limitations

This is a method and protocol reproduction on public substitute data. It does not claim numerical reproduction of the original Flickr/YouTube to Kodak/CCV/HMDB51 tables, HMDB51 accuracy, natural cross-dataset transfer, original hardware timing, or a universal SOTA result. The original source-domain inputs are not available in the required four-MAT format.

The primary saved evidence covers the core MIL/PI/DA families and representative baselines. Optional exploratory modules remain in `experiments/` and `configs/`, but they are not silently promoted into the final result tables.

## Citation

```bibtex
@article{niu2016privileged,
  author = {Niu, Li and Li, Wen and Xu, Dong},
  title = {Exploiting Privileged Information from Web Data for Action and Event Recognition},
  journal = {International Journal of Computer Vision},
  year = {2016},
  doi = {10.1007/s11263-015-0862-5}
}

@inproceedings{xu2016msrvtt,
  author = {Xu, Jun and Mei, Tao and Yao, Ting and Rui, Yong},
  title = {MSR-VTT: A Large Video Description Dataset for Bridging Video and Language},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
  year = {2016}
}

@inproceedings{girdhar2023imagebind,
  author = {Girdhar, Rohit and El-Nouby, Alaaeldin and Liu, Zhuang and Singh, Mannat and Alwala, Kalyan Vasudev and Joulin, Armand and Misra, Ishan},
  title = {ImageBind: One Embedding Space To Bind Them All},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year = {2023}
}

@misc{richyhu999features,
  author = {{RichyHu999}},
  title = {Multimodal Teacher Data: MSR-VTT Feature Mirror},
  howpublished = {Hugging Face dataset repository},
  url = {https://huggingface.co/datasets/RichyHu999/multimodal-teacher-data/tree/bf56d4f6b7a191b393e50894b81ee4089a4131f4},
  note = {Fixed repository revision bf56d4f6b7a191b393e50894b81ee4089a4131f4; upstream feature extraction settings were not independently verified}
}
```
