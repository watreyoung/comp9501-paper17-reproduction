# Final public-data reproduction results

This page reports the **saved outputs of the completed experiments** in this repository. The numbers below are read from `results/core/summary.json`, `results/baselines/summary.json`, and `results/literature/summary.json`; they are not illustrative values and no new training is performed when viewing the repository.

The results correspond to the reproduction section of the COMP9501 Paper 17 presentation. They use the public MSR-VTT feature mirror, ImageBind visual features, source-training captions, five seeds, 0/20/40% positive-bag contamination, five positive and five negative bags of five instances, and validation-only parameter selection. Values below are five-seed mean mAP with sample standard deviation in percentage points.

## Core MIL/PI/DA results at 20% contamination

### Full setting

| Method | mAP (%) | sample SD (pp) |
| --- | ---: | ---: |
| SVM | 38.23 | 1.16 |
| pSVM+ | 38.23 | 1.16 |
| sMIL | 44.63 | 0.51 |
| sMIL-PI | 44.63 | 0.51 |
| sMIL-PI-DA | 44.63 | 0.51 |
| mi-SVM | 42.59 | 1.70 |
| mi-SVM-PI | 44.36 | 1.32 |
| mi-SVM-PI-DA | 39.10 | 0.50 |
| MIL-CPB | 23.50 | 2.05 |
| MIL-CPB-PI | 28.72 | 2.44 |
| MIL-CPB-PI-DA | 39.28 | 0.58 |
| sMIL-PI-full | 44.70 | 0.61 |
| sMIL-PI-shuffled | 44.63 | 0.51 |
| mi-SVM-PI-full | 30.06 | 2.87 |
| mi-SVM-PI-shuffled | 44.09 | 1.29 |
| MIL-CPB-PI-full | 36.28 | 3.59 |
| MIL-CPB-PI-shuffled | 29.63 | 1.94 |
| KMM+sMIL-PI | 44.95 | 0.28 |
| KMM+mi-SVM-PI | 34.22 | 2.29 |
| KMM+MIL-CPB-PI | 28.85 | 2.63 |

### Visual-cluster setting

| Method | mAP (%) | sample SD (pp) |
| --- | ---: | ---: |
| SVM | 31.31 | 2.70 |
| pSVM+ | 31.31 | 2.70 |
| sMIL | 33.98 | 2.63 |
| sMIL-PI | 33.98 | 2.63 |
| sMIL-PI-DA | 34.01 | 2.61 |
| mi-SVM | 29.51 | 2.50 |
| mi-SVM-PI | 32.52 | 2.45 |
| mi-SVM-PI-DA | 31.31 | 2.95 |
| MIL-CPB | 18.89 | 3.30 |
| MIL-CPB-PI | 20.31 | 2.85 |
| MIL-CPB-PI-DA | 31.31 | 2.95 |
| sMIL-PI-full | 34.04 | 2.66 |
| sMIL-PI-shuffled | 33.98 | 2.63 |
| mi-SVM-PI-full | 26.27 | 2.61 |
| mi-SVM-PI-shuffled | 31.80 | 2.24 |
| MIL-CPB-PI-full | 25.74 | 2.82 |
| MIL-CPB-PI-shuffled | 20.87 | 2.65 |
| KMM+sMIL-PI | 32.29 | 2.66 |
| KMM+mi-SVM-PI | 29.28 | 1.87 |
| KMM+MIL-CPB-PI | 14.29 | 0.91 |

## Representative baseline results at 20% contamination

### Full setting

| Method | mAP (%) | sample SD (pp) |
| --- | ---: | ---: |
| SA+SVM | 17.81 | 1.75 |
| SA+sMIL-PI | 30.33 | 1.74 |
| TCA+SVM | 21.04 | 1.11 |
| TCA+sMIL-PI | 31.28 | 0.93 |
| KCCA+SVM | 31.59 | 2.50 |
| SVM-2K | 39.32 | 0.72 |

### Visual-cluster setting

| Method | mAP (%) | sample SD (pp) |
| --- | ---: | ---: |
| SA+SVM | 13.42 | 2.47 |
| SA+sMIL-PI | 22.73 | 1.39 |
| TCA+SVM | 15.76 | 1.44 |
| TCA+sMIL-PI | 25.31 | 2.00 |
| KCCA+SVM | 26.02 | 3.43 |
| SVM-2K | 31.61 | 3.03 |

## Verified literature-baseline results at 20% contamination

### Full setting

| Method | mAP (%) | sample SD (pp) |
| --- | ---: | ---: |
| KMM+SVM | 38.53 | 1.39 |
| STM | 38.22 | 1.10 |
| RT | 35.33 | 1.32 |
| DASVM | 37.10 | 1.25 |
| SVM-uniform | 44.09 | 0.92 |

### Visual-cluster setting

| Method | mAP (%) | sample SD (pp) |
| --- | ---: | ---: |
| KMM+SVM | 30.01 | 3.40 |
| STM | 30.58 | 3.17 |
| RT | 28.21 | 2.23 |
| DASVM | 16.05 | 0.74 |
| SVM-uniform | 33.85 | 2.58 |

## Interpretation

- PI and DA gains depend on the MIL family. CPB improves strongly with joint DA; sMIL is nearly unchanged; mi-SVM PI helps in the primary setting but its DA variant can decrease.
- Correctly paired captions do not consistently outperform shuffled captions, so a PI gain alone does not establish a semantic-pairing explanation.
- Partial versus full PI is method-dependent.
- These are controlled substitute-data results. They do not reproduce the original Flickr/YouTube to Kodak/CCV/HMDB51 numerical tables, HMDB51 accuracy, natural cross-dataset transfer, or a universal SOTA claim.

## Audit provenance

The core batch contains 9,600 test fits and 1,056 validation candidates; the baseline extension contains 576 validation fits and 2,880 test fits. Twenty failed core validation candidates were retained and excluded from selection. The corresponding audit records are [`core/delivery_audit.json`](core/delivery_audit.json) and [`baselines/delivery_audit.json`](baselines/delivery_audit.json).
