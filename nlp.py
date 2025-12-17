import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from transformers import pipeline

nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

text = input("Write a sentence: ")
text.lower()
text.translate(str.maketrans("","", string.punctuation))
tokens = word_tokenize(text)
print(tokens)
tagged_tokens = nltk.pos_tag(tokens)
print(tagged_tokens)

import spacy
nlp = spacy.load('en_core_web_sm')
doc = nlp(text)
for ent in doc.ents:
    print(ent.text, ent.label_)

sentiment_analyzer = pipeline('sentiment-analysis')
result = sentiment_analyzer(text)
print(f"Label: {result[0]['label']}, Result: {result[0]['score']:.2f}")

summarizer = pipeline('summarization')
summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
print(summary)