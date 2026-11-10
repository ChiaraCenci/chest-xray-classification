# Chest X-ray Classification

A university deep learning project distinguishing COVID, Normal and Viral Pneumonia radiographs with a CNN trained from scratch.

## Data and model
The original local subset of the [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) contains 9,961 images (3,616 COVID, 5,000 Normal and 1,345 Viral Pneumonia). Data are not included. The original selection of Normal images is undocumented; downloading the full database does not reconstruct this subset.

Grayscale images are resized to 224 × 224, normalised and contrast-adjusted (1.15). Training uses horizontal flips, rotation ±10.8° and zoom 0.05. Four convolutional blocks (32/64/128/256 filters) use batch normalisation, ReLU and pooling, followed by global average pooling, dropout and three-class softmax. Optimisation uses AdamW and balanced class weights.

## Evaluation and saved results
Nested stratified cross-validation uses three outer and three inner folds. Learning rate and batch size are selected by inner macro F1. Each final model uses 80% of its outer-training partition for fitting and 20% for validation and early stopping. Outer tests are excluded from tuning.

| Metric | Outer-fold mean ± sample SD |
|---|---|
| Macro precision | 0.8364 ± 0.0134 |
| Macro recall | 0.8678 ± 0.0168 |
| Macro F1 | 0.8465 ± 0.0173 |

Pooled outer-test accuracy is 0.8406. These are historical saved results, not a new reproduction. The notebook includes fold metrics, pooled and per-class reports, and authentic original figures in `assets/results/`.

## Run
1. Create a Python 3.10 or 3.11 environment and install `pip install -r requirements.txt`.
2. Obtain the dataset separately, following the source terms and citation instructions. Arrange images under `COVID/images`, `Normal/images`, and `Viral Pneumonia/images`.
3. Set `CHEST_XRAY_DATASET` or edit the single `DATASET_ROOT` setting in the notebook. Colab users can uncomment the Drive mount lines.
4. Open `Chest_Xray_Classification.ipynb` in JupyterLab or Colab and run cells in order. A GPU is recommended; the full protocol trains 39 models.

The proposed dependency ranges are not an exact environment lock. Original library versions and a file manifest were not saved; seeds alone do not guarantee identical results. Review [validation and changes](VALIDATION.md) before running or publishing.

## Limitations
Splits are stratified at image level. Patient independence and absence of duplicate images have not been established. No external or clinical validation is available. Acquisition-source confounding and local subset selection limit generalisation. Three-fold variability is not a confidence interval. No code licence has been selected.
