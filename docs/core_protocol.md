# Core public protocol

- Dataset: MSR-VTT public feature mirror; ImageBind visual features and captions.
- Settings: `full` and constructed `visual_cluster`.
- Bag construction: five positive and five negative bags, five instances per bag, contamination 0.0/0.2/0.4.
- Seeds: 0 through 4.
- Model families: SVM/pSVM+, sMIL, mi-SVM, and MIL-CPB, with no-PI, PI, DA, full-PI, shuffled-text, and KMM controls where defined.
- Tuning: regularization is selected on validation only; test labels are used only for final AP calculation.
- Reporting: classwise AP is macro-averaged; seed mean and sample standard deviation are reported.
- Boundary: this protocol validates controlled method trends on substitute data and does not reproduce the original paper's data tables.
