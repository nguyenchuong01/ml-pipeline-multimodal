# Measured runtimes (Google Colab, one *Run all* each)

| Notebook | Runtime type | Total time | Notes |
|---|---|---|---|
| 01_tabular | CPU | about 42 min (estimated from file timestamps) | 5-fold CV, 10 configurations x 4 models; Random Forest is the slowest model (about 29 s per fit on average) |
| 02_text | T4 GPU | 47 min (18:46 to 19:33) | MeCab tokenisation 35 s; TF-IDF/BoW grid about 33 min; e5-base embeddings grid about 12 min; slowest: Random Forest, char n-gram TF-IDF, linear SVM on embeddings |
| 03_image | T4 GPU | 27 min (09:29 to 09:56) | feature extraction for 7 backbone settings 570 s in total; classifier grid about 16 min; includes the 1.1 GB CUB download |

All three notebooks ran without errors (0 failed configurations) with `QUICK = False`. The times exclude downloading the result files from Colab.
