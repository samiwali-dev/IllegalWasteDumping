Archived exploratory protocol (v1)
==================================

These are the first SVM and CNN-LSTM outputs. Their checkpoints were selected
using validation F1 for the Dumping class alone. That rule rewarded the
CNN-LSTM checkpoint that predicted Dumping for every validation and test video.
The corresponding model files are in `models/protocol_v1_positive_f1/`.

This archive is kept for provenance. It is not part of the revised four-model
comparison, which selects checkpoints using validation macro F1. Because the
official test results of v1 were inspected before this revision, the revised
comparison must be described as exploratory rather than a pristine one-shot
holdout evaluation.

The old JSON files retain the absolute paths they had before being archived;
use the paths above to locate the archived model files.
