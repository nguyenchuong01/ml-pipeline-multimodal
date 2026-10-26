# Dataset cards

## Task 1 - OkCupid profiles (JSE)
* **Source:** https://github.com/rudeboybert/JSE_OkCupid (`profiles_revised.csv.zip`), Kim & Escobedo-Land (2015), *OkCupid Profile Data for Introductory Statistics and Data Science Courses*, Journal of Statistics Education 23(2).
* **Size:** 59,946 rows, 19 columns; 38,318 rows after target construction (STEM 24.9 percent).
* **Why it fits:** real missing values (offspring 59 %, diet 41 %, religion 34 %), disguised missing (`income = -1`, 81 %), many categorical and compound fields (`ethnicity`, `speaks`).
* **Target:** `job` in {science / tech / engineering, computer / hardware / software} = STEM; other known fields = non-STEM. Excluded: missing job, `other`, `student`, `rather not say`, `unemployed`, `retired`. `job` is dropped from features (label leakage).
* **Ethics:** public profiles, released by the authors for teaching on the condition that the data stay public; analysed in aggregate, no re-identification attempt.
* **Limitations:** volunteer sample, San Francisco area, self-reported labels.

## Task 2 - livedoor News Corpus
* **Source:** https://www.rondhuit.com/download.html (`ldcc-20140209.tar.gz`, about 8.9 MB).
* **Size:** 7,367 articles, 9 categories (511 to 900 per class; imbalance ratio about 1.76).
* **Format:** one `.txt` per article: line 1 URL, line 2 date, line 3 title, rest body.
* **License:** CC BY-ND 2.1 JP; attribution requirements vary per category (`LICENSE.txt` inside each folder). Texts are neither committed nor redistributed.
* **Tokenisation:** MeCab through `fugashi` + `unidic-lite`.
* **Alternative considered:** UIT-VSFC (`uitnlp/vietnamese_students_feedback`), kept as fallback if the instructor requires a Vietnamese dataset.

## Task 3 - CUB-200-2011
* **Source:** CaltechDATA https://data.caltech.edu/records/20098 (`CUB_200_2011.tgz`, 1.15 GB, md5 `97eceeb196236b17998738112f37df78`).
* **Size:** 11,788 images, 200 species; official split 5,994 train / 5,794 test; bounding boxes and part annotations provided.
* **Limitation:** overlap with ImageNet (stated on the dataset page), so ImageNet-pretrained backbones may have seen test images.
