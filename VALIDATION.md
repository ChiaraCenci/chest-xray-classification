# Validation and changes

The original notebook and supporting Word notes were read without modification. Code takes precedence where the notes describe full outer-training retraining or stopping patience differently.

Preserved: CNN layers, preprocessing, augmentation factors, seeds, mixed precision, search grid, folds, class weighting, callbacks and epoch budgets. Changes: English documentation and console labels, a configurable dataset root, optional Colab mounting, an empty-dataset guard, and correction of the misleading full-training comment and rotation-angle comment. The radiograph preview is omitted. Historical text outputs and seven figure images are presented separately from executable cells. The pooled figure retains its original Italian title.

Checks completed: JSON notebook structure and code-cell Python syntax; saved summary values checked against original outputs; embedded figures extracted from original PNG bytes; local figure references verified. No original images, personal mandatory Drive paths or credentials are included. Dependency ranges are proposed, not tested as a clean installation or recovered from the original run.

Not completed: full notebook execution, GPU training, dataset checks, duplicate-content detection, patient overlap checks, exact subset reconstruction, or reproduction of historical predictions. The dataset was not supplied. TensorFlow import availability was probed separately; this does not verify end-to-end compatibility. Review the original subset selection and decide on a code licence before publication. No GitHub publication was performed.
