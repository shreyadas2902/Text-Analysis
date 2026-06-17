# Text Preprocessing

import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_and_lemmatize(text):
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return tokens


def preprocess_dataframe(df, text_column):
    df = df.dropna(subset=[text_column]).copy()
    df["cleaned"] = df[text_column].apply(clean_text)
    df["tokens"] = df["cleaned"].apply(tokenize_and_lemmatize)
    return df


if __name__ == "__main__":
    # Example with sample data
    sample = pd.DataFrame({
        "review": [
            "This product is absolutely amazing, highly recommend!",
            "Terrible quality, broke after one day. Very disappointed.",
            "It's okay I guess, nothing special about it.",
        ]
    })

    result = preprocess_dataframe(sample, "review")
    print(result[["review", "cleaned", "tokens"]])
