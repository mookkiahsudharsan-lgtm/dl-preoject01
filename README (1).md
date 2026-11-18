# Sentiment Analysis on Movie Reviews (Naive Bayes)

A Jupyter notebook that classifies movie reviews as **positive** or **negative** using a Multinomial Naive Bayes classifier, and compares two text feature representations: Bag of Words (`CountVectorizer`) and TF-IDF (`TfidfVectorizer`).

## Overview

The notebook uses the NLTK `movie_reviews` corpus (1,000 positive and 1,000 negative reviews) and walks through the full pipeline:

1. Load the corpus from NLTK and inspect its categories (`neg`, `pos`).
2. Convert each tokenized review back into a single string (token join with light punctuation cleanup; a Moses detokenizer is also demonstrated).
3. Create labels: `0` for negative reviews and `1` for positive reviews.
4. Vectorize the text with **CountVectorizer** (Bag of Words) and with **TfidfVectorizer**. Both use `min_df=2`, lowercase, and English stop-word removal, giving 23,784 features.
5. Split the data into training (75%, 1,500 reviews) and testing (25%, 500 reviews) sets with `random_state=5`.
6. Train a `MultinomialNB` classifier on each representation and evaluate with accuracy and a confusion matrix.

## Results

| Features | Accuracy | Correctly classified | Misclassified |
|----------|----------|----------------------|---------------|
| Bag of Words (CountVectorizer) | 0.798 | 399 | 101 |
| TF-IDF (TfidfVectorizer) | 0.820 | 410 | 90 |

TF-IDF features gave a modest improvement over raw word counts on this split.

> Note: in the notebook's confusion-matrix code, class `0` (negative) is treated as the "positive" case when naming TP/FP/FN/TN. Interpret those labels accordingly.

## Requirements

- Python 3.6+
- pandas
- numpy
- nltk
- scikit-learn
- Jupyter Notebook

Install dependencies:

```bash
pip install pandas numpy nltk scikit-learn notebook
```

Download the required NLTK data:

```python
import nltk
nltk.download('movie_reviews')
nltk.download('punkt')
```

## Usage

```bash
jupyter notebook Sentiment_Analysis.ipynb
```

Run the cells in order from top to bottom.

## Known Compatibility Issues

The notebook was written against older library versions. On current versions you may need these changes:

- `from nltk.tokenize.moses import MosesDetokenizer` was removed from NLTK. Use `sacremoses` (`from sacremoses import MosesDetokenizer`) instead, or skip it, since the notebook's actual pipeline uses `" ".join(...)`.
- `np.int` was removed in NumPy 1.24+. Use `dtype=int` instead.
- `CountVectorizer.get_feature_names()` was removed in newer scikit-learn. Use `get_feature_names_out()`.
- Printing `rev_list` in full can exceed Jupyter's IOPub data rate limit; print a single item (e.g. `rev_list[0]`) instead.

## Possible Improvements

- Use `TfidfVectorizer` with n-grams (e.g. `ngram_range=(1, 2)`).
- Keep the sparse matrices instead of converting to dense DataFrames to save memory.
- Use cross-validation instead of a single train/test split.
- Compare with other classifiers such as Logistic Regression or Linear SVM.

## Project Structure

```
.
├── Sentiment_Analysis.ipynb
└── README.md
```
