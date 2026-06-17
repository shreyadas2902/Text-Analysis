# Sentiment Classification using BERT

import pandas as pd
from transformers import pipeline


def load_classifier(model_name="distilbert-base-uncased-finetuned-sst-2-english"):
    return pipeline("sentiment-analysis", model=model_name)


def classify_sentiment(texts, classifier):
    results = classifier(texts, truncation=True, max_length=512)
    return [
        {"text": t, "label": r["label"], "score": round(r["score"], 4)}
        for t, r in zip(texts, results)
    ]


def run(texts):
    classifier = load_classifier()
    predictions = classify_sentiment(texts, classifier)

    df = pd.DataFrame(predictions)
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    examples = [
        "The product exceeded my expectations — great quality!",
        "Terrible experience, wouldn't recommend to anyone.",
        "It's decent for the price, nothing extraordinary.",
    ]
    run(examples)
