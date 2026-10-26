# ML Pipeline Multimodal: tabular, text and image classification

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](requirements.txt)
[![Tests](https://img.shields.io/badge/tests-pytest-green.svg)](tests/)
[![Colab](https://img.shields.io/badge/run%20on-Google%20Colab-F9AB00.svg)](#run-on-google-colab)

Config-driven, leakage-safe machine learning pipelines for three data modalities, built as the big assignment of the **Machine Learning** course. Every experiment is described in a YAML file; the notebooks only load a config and call reusable modules.

| Task | Dataset | Problem | Features | Classifiers |
|---|---|---|---|---|
| 1. Tabular | OkCupid profiles (59,946 rows, 19 columns, missing + categorical) | STEM vs non-STEM occupation | imputation, one-hot / frequency / ordinal / multi-hot, scaling, PCA | Logistic Regression, linear SVM, RBF SVM, Random Forest |
| 2. Text | livedoor News Corpus (7,367 Japanese articles, 9 topics) | topic classification | BoW, TF-IDF (MeCab words, character n-grams), `multilingual-e5-base` embeddings | LR, linear SVM, Complement NB, Random Forest, RBF SVM |
| 3. Image | CUB-200-2011 (11,788 images, 200 bird species) | fine-grained species classification | frozen ResNet50 / EfficientNet / ViT / DINOv2 embeddings, HOG baseline | LR, linear SVM (SGD), Random Forest |

## Course information

| | |
|---|---|
| Course | Machine Learning (Học máy), CO3117 |
| University | Ho Chi Minh City University of Technology (HCMUT - VNU-HCM) |
| Semester | Semester 1, 2026-2027 |
| Instructor (GVHD) | Trương Vĩnh Lân |
| Student | Phạm Trần Nguyên Chương, ID 2352142, chuong.pham0101@hcmut.edu.vn (individual project) |

## Goals

1. Build a **reproducible**, config-driven ML pipeline: fixed seeds, pinned dependencies, no manual steps.
2. Prevent **data leakage**: every preprocessing step is fitted inside the cross-validation pipeline on the training fold only.
3. Compare **traditional features vs pretrained embeddings** with honest metrics (macro-F1, mean ± std over StratifiedKFold, fit / predict time) and written analysis.
4. Make the project runnable end to end on Google Colab with **Run all**.

## Run on Google Colab

Open a notebook and choose **Runtime > Run all** (select a GPU runtime for notebooks 2 and 3). Each notebook clones this repository, installs its few extra dependencies, downloads its dataset from the public source (no login, no personal cloud mounting) and writes figures, tables and feature files.

| Notebook | Open |
|---|---|
| 01 Tabular | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nguyenchuong01/ml-pipeline-multimodal/blob/main/notebooks/01_tabular.ipynb) |
| 02 Text | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nguyenchuong01/ml-pipeline-multimodal/blob/main/notebooks/02_text.ipynb) |
| 03 Image | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nguyenchuong01/ml-pipeline-multimodal/blob/main/notebooks/03_image.ipynb) |

Feature extraction (notebooks 2 and 3, GPU) and the 5-fold grids dominate the runtime; extracted features are cached as `.npy`, so a second *Run all* in the same session is much faster. Set `QUICK = True` in the config cell for a fast smoke test. (Measured runtimes are listed in `docs/runtime.md` once the notebooks have been run on Colab.)

## Run locally

```bash
git clone https://github.com/nguyenchuong01/ml-pipeline-multimodal.git
cd ml-pipeline-multimodal
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest                      # unit tests on synthetic data, no downloads
jupyter lab notebooks/
```

## Repository structure

```
.
├── notebooks/          # 01_tabular, 02_text, 03_image (Colab front-end; thin, call modules)
├── modules/            # reusable code
│   ├── config.py  utils.py  models.py  evaluation.py  viz.py
│   ├── tabular/        # data cleaning, configurable preprocessing, experiment grid
│   ├── text/           # corpus loader, MeCab tokenizer, TF-IDF / embeddings, experiment grid
│   └── image/          # CUB loader, timm feature extractors, HOG, experiment grid
├── configs/            # tabular.yaml, text.yaml, image.yaml (all experiment settings)
├── features/           # generated .npy feature files (git-ignored, included in the submission zip)
├── reports/            # report PDF, figures/, results/ (CV tables as CSV)
├── docs/               # dataset cards and licences, meeting notes
├── scripts/            # notebook builders and the submission packager
└── tests/              # pytest suite (cleaning, leakage-safe preprocessing, parsers, extractors)
```

## Method highlights

* **Leakage control.** Imputers, encoders, scalers, PCA, TF-IDF vocabularies and IDF weights live inside one `sklearn.Pipeline` with the classifier. The label source column `job` is removed from the tabular features. Tokenisation (stateless) is the only step done before splitting.
* **Disguised missing values.** `income = -1` is treated as missing (81 percent of rows) and compared against dropping high-missing columns and plain imputation.
* **One-factor-at-a-time ablations** (`experiments:` blocks in YAML) make every comparison attributable to a single change.
* **Imbalance-aware metrics.** macro-F1 is the primary metric, with class weights where the data are imbalanced.
* **Failure isolation.** A failing configuration is recorded in the results table instead of aborting *Run all*.

## Results

The final tables and figures are generated by the notebooks into `reports/results/` and `reports/figures/`. The analysis is in the report (`reports/report.pdf`).

| Task | Best configuration (CV) | macro-F1 | Notes |
|---|---|---|---|
| 1 Tabular | _fill after running_ | | |
| 2 Text | _fill after running_ | | |
| 3 Image | _fill after running_ | | |

## Data sources, licences and limitations

Raw data are downloaded at run time and are **not** stored in this repository. See [docs/datasets.md](docs/datasets.md).

* OkCupid profiles: Kim & Escobedo-Land (2015), *Journal of Statistics Education*. Public profiles released by the authors for teaching; the data must remain public and are analysed in aggregate only.
* livedoor News Corpus: RONDHUIT Co., Ltd., CC BY-ND 2.1 JP. Texts are not redistributed here.
* CUB-200-2011: Wah et al. (2011), CaltechDATA. Images overlap with ImageNet, which may inflate results for ImageNet-pretrained backbones.

## Reproducibility and submission

Seeds are fixed in each config (`seed: 42`). `python scripts/make_submission.py` creates the course zip with `notebooks/`, `modules/`, `reports/` and `features/`.

## License

Code: MIT (see `LICENSE`). Datasets keep their original licences.
# ml-pipeline-multimodal
