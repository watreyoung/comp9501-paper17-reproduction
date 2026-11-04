# Reproduction coverage

This inventory describes the evidence included in the repository for the reproduction section of the COMP9501 presentation. It distinguishes saved final results from source modules that were not used for the final comparison.

## Completed saved evidence

| Reproduction item | Repository evidence | Coverage |
| --- | --- | --- |
| Public substitute data and fixed protocol | [`docs/data_search.md`](../docs/data_search.md), [`docs/core_protocol.md`](../docs/core_protocol.md), [`configs/core.json`](../configs/core.json) | MSR-VTT with ImageBind visual features and source-training captions; full and visual-cluster settings; five seeds; 0%, 20%, and 40% positive-bag contamination; validation-only selection |
| Core MIL comparisons | [`experiments/core_run.py`](../experiments/core_run.py), [`results/core/summary.json`](core/summary.json) | SVM/pSVM+, sMIL, mi-SVM, and MIL-CPB |
| Privileged-information comparisons | [`results/core/REPORT.md`](core/REPORT.md), [`results/core/EVIDENCE.md`](core/EVIDENCE.md) | Partial PI, DA, full PI, and shuffled-text controls |
| Representative feature and kernel baselines | [`experiments/baseline_extension.py`](../experiments/baseline_extension.py), [`results/baselines/summary.json`](baselines/summary.json) | SA, TCA, KCCA, and SVM-2K heads |
| Literature baseline batch | [`experiments/literature_run.py`](../experiments/literature_run.py), [`results/literature/summary.json`](literature/summary.json) | KMM+SVM, STM, RT, DASVM, and SVM-uniform |
| Saved numerical results | [`results/final_report.md`](final_report.md), [`results/summary.json`](summary.json) | Five-seed mAP tables at all three contamination levels for both public settings |
| Independent result checks | [`results/core/delivery_audit.json`](core/delivery_audit.json), [`results/baselines/delivery_audit.json`](baselines/delivery_audit.json) | AP recomputation, task and bag identities, source/input hashes, transform checks, and validation-based selection |

The saved results are the outputs used for the presentation tables. They are not illustrative values. The core batch contains 1,056 validation fits and 9,600 test fits, with 20 failed validation candidates retained and excluded from selection. The representative-baseline extension contains 576 validation fits and 2,880 test fits. The literature-baseline batch contains 480 validation fits and 2,400 test fits. Across all three batches, this is 16,992 fits and 14,880 successful test fits; the saved batch audits pass for all completed test outputs.

## Main conclusions represented in the presentation

- PI and DA effects depend on the MIL family: sMIL changes little, mi-SVM PI helps in the main setting while its DA variant can decrease, and MIL-CPB improves strongly with DA.
- Correctly paired captions do not consistently outperform shuffled captions under this public substitute protocol.
- Partial versus full PI is method-dependent.
- The results support controlled method and trend validation. They do not reproduce the original Flickr/YouTube to Kodak/CCV/HMDB51 numerical tables, HMDB51 accuracy, natural cross-dataset transfer, or a universal SOTA claim.

## Implemented but excluded from final evidence

The repository retains source modules and tests for several exploratory branches, but no completed result from them is included in the final tables: GFK, DIP, SGF, the specified MILBoost variant, full MIMLBoost, Classeme, and MIMLSVM event evaluation. The presentation also excludes these branches, together with the original-data experiments and original hardware timing claims.

Keeping these branches separate prevents an implementation or a partial run from being mistaken for a completed reproduction result.
