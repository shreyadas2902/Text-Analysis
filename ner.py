#using spaCy

import spacy
import pandas as pd

nlp = spacy.load("en_core_web_sm")


def extract_entities(text):
    doc = nlp(text)
    return [{"entity": ent.text, "label": ent.label_, "description": spacy.explain(ent.label_)} for ent in doc.ents]


def run(texts):
    all_entities = []
    for text in texts:
        entities = extract_entities(text)
        for e in entities:
            e["source"] = text[:60] + "..."
        all_entities.extend(entities)

    df = pd.DataFrame(all_entities)
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    examples = [
        "Elon Musk announced that Tesla will open a new factory in Austin, Texas next year.",
        "Apple released the iPhone 15 in September 2023, priced at $799.",
        "The United Nations held a summit in Geneva to discuss climate change.",
    ]
    run(examples)
