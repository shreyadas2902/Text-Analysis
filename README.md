# Text Analysis Pipeline

End-to-end NLP pipeline for sentiment analysis, named entity recognition (NER), and topic modeling on unstructured text data.

**Dataset:** Kaggle  
**Tools:** Python, BERT, spaCy, NLTK, Gensim, Hugging Face Transformers

## Project Structure
```
text-analysis/
├── src/
│   ├── preprocessing.py       # Text cleaning and tokenization
│   ├── sentiment.py           # BERT-based sentiment classification
│   ├── ner.py                 # Named entity recognition with spaCy
│   └── topic_modeling.py      # LDA topic modeling with Gensim
├── notebooks/
│   └── analysis.ipynb         # Full walkthrough notebook
├── data/                      # Place your Kaggle dataset here
└── requirements.txt
```

## Setup
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Run
```bash
python src/preprocessing.py
python src/sentiment.py
python src/ner.py
python src/topic_modeling.py
```
