# Emotion-Focused Analysis of Stock Tweets

![image](https://github.com/user-attachments/assets/ebc8d5e5-3140-48f7-8aa6-ba2f23e206a2)

This repository contains the code to perform multi-class emotion classification and topic modeling on stock-related tweets using various NLP techniques.

[Link to the StockEmotion Dataset](https://github.com/adlnlp/StockEmotions).

![image](https://github.com/user-attachments/assets/b3a60d77-30d4-46c3-9707-5055b0a27960)

## Project Summary

We investigate:
- Emotion classification using TF-IDF, Word2Vec, and contextual embeddings (BERTweet, Distil-RoBERTa)
- Topic modeling with LDA and BERTopic
- Integration of emoji features and sentiment lexicons (VADER, NRC, Bing Liu)

For full details, see the [project report](./TM_report_Smith_Verga.pdf).

## Notebooks Overview

| Notebook                              | Description                                                                 |
|--------------------------------------|-----------------------------------------------------------------------------|
| `dataset_exploration.ipynb`          | Exploratory Data Analysis (EDA)                                            |
| `TM_preproc_FE_classification.ipynb` | Preprocessing, feature extraction, training of classifiers                 |
| `word2vec_multi.ipynb`               | Word2Vec training and downstream classification                            |
| `BERT_embeddings_and_classifier.ipynb`| Embedding generation (BERTweet, Distil-RoBERTa) + classifier               |
| `topic_modeling_LDA.ipynb`           | LDA-based topic modeling                                                   |
| `topic_modeling_BERTopic.ipynb`      | BERTopic-based topic modeling                                              |

**Note**: Pre-trained models (e.g., Word2Vec) and large files are excluded from the repo.

## Environment Setup

This project uses **Python 3.12**. It's recommended to create a virtual environment to manage dependencies.

### Create and activate a virtual environment (venv)

**Unix/macOS:**
```bash
python3.12 -m venv venv
source venv/bin/activate
````

**Windows:**

```bash
py -3.12 -m venv venv
venv\Scripts\activate
```

### Install dependencies

Start by upgrading pip:

```bash
pip install --upgrade pip
```

Install required packages manually:

```bash
pip install bertopic gensim nltk spacy emoji vaderSentiment
```

Additional packages may be required based on notebook usage (e.g., `scikit-learn`, `matplotlib`, `pandas`, `umap-learn`).

### Set up NLTK and spaCy

After installing:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')

import spacy
spacy.cli.download("en_core_web_sm")
```

## External Files Needed

* **Lexicons**: Bing Liu and NRC Emotion Lexicons (update paths accordingly).
* **Processed CSV**: Required before running `topic_modeling_BERTopic.ipynb`.
* **Pre-trained Word2Vec**: Must be trained separately or loaded from local storage.

## Findings

* **Best classifier**: XGBoost with Bigram TF-IDF (macro-F1 ≈ 0.32)
* **Topic modeling coherence**: \~0.3, indicating challenges due to tweet brevity/noise
* **Emoji features** proved critical in both classification and topic modeling

## Authors

* **Robin Smith**
* **Sergio Verga**

*Università degli Studi di Milano-Bicocca*

## References

* [BERTweet](https://aclanthology.org/2020.emnlp-demos.2.pdf)
* [VADER](https://ojs.aaai.org/index.php/ICWSM/article/view/14550)
* [NRC Emotion Lexicon](https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm)
* [Bing Liu's Lexicon](https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html)

> For methodology and evaluation, see: [`TM_report_Smith_Verga.pdf`](./TM_report_Smith_Verga.pdf).
