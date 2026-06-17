# Topic Modeling using LDA (Gensim)

import pandas as pd
from gensim import corpora, models
from preprocessing import clean_text, tokenize_and_lemmatize


def build_lda_model(tokenized_docs, num_topics=3, passes=15):
    dictionary = corpora.Dictionary(tokenized_docs)
    dictionary.filter_extremes(no_below=2, no_above=0.9)

    corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]
    lda = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=passes, random_state=42)

    return lda, dictionary, corpus


def print_topics(lda, num_words=6):
    print("\nDiscovered Topics:")
    print("-" * 40)
    for idx, topic in lda.print_topics(num_words=num_words):
        print(f"Topic {idx + 1}: {topic}\n")


def get_doc_topics(lda, corpus):
    return [max(lda[doc], key=lambda x: x[1]) for doc in corpus]


if __name__ == "__main__":
    raw_docs = [
        "Machine learning models require large datasets for training and validation.",
        "Deep learning neural networks are used in image recognition tasks.",
        "Stock market prices fluctuate based on investor sentiment and economic indicators.",
        "Investment portfolios should be diversified to minimize financial risk.",
        "Patients with chronic illness require long-term treatment and care management.",
        "Medical research focuses on developing new drugs and therapies for disease.",
    ]

    tokenized = [tokenize_and_lemmatize(clean_text(doc)) for doc in raw_docs]
    lda, dictionary, corpus = build_lda_model(tokenized, num_topics=3)

    print_topics(lda)

    assignments = get_doc_topics(lda, corpus)
    for doc, (topic_id, prob) in zip(raw_docs, assignments):
        print(f"[Topic {topic_id + 1} | {prob:.2f}] {doc[:60]}...")
