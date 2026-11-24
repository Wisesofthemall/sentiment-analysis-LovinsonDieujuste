# NLP Application: Sentiment Analysis

**COP4020 – Programming Languages** · Lovinson Dieujuste

Sentiment analysis of Amazon Alexa customer reviews (`customer_reviewers.tsv`) using an LSTM neural network built with Keras/TensorFlow.

- **Notebook:** `sentiment_analysis-LovinsonDieujuste.ipynb`
- **Run it in Google Colab:** https://colab.research.google.com/drive/1eszaUbzGVuUmbipLK5f5OsqZUZwYmGWT
- **Dependencies:** `requirements.txt`

## What it does
1. **Data wrangling** – loads the TSV into a DataFrame, prints the first ten rows, checks for and removes missing values, removes stop words, and draws a word cloud.
2. **Data engineering** – encodes labels with `LabelEncoder` and turns reviews into padded integer sequences with `Tokenizer`.
3. **Model** – Embedding(500, 120) → SpatialDropout1D(0.4) → LSTM(176, dropout=0.2, recurrent_dropout=0.2) → Dense(2, softmax); 269,442 trainable parameters.
4. **Training / evaluation** – 80/20 split, 5 epochs, batch size 32, then a `classification_report` on the test set (~90% accuracy).
5. **Client program** – prompts for a product review and prints the predicted sentiment.

## How to run
Open the notebook in Colab, upload `customer_reviewers.tsv` when prompted (or into the Files panel), then choose **Runtime → Run all**.
