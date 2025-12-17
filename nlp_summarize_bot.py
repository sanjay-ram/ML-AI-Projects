import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from transformers import pipeline

text = input("You: ")
text.lower()
text.translate(str.maketrans("","", string.punctuation))
tokens = word_tokenize(text)
taggend_tokens = nltk.pos_tag(tokens)

nlp = spacy.load("de_core_news_sm")
doc = nlp(text)
for ent in doc.ents:
    print(ent.text, ent.label_)

sentiment_analyser = pipeline('sentiment-analysis')
result = sentiment_analyser(text)
summarizer = pipeline('summarization')
summary = summarizer(text, max_length=50, min_length=50, do_sample=False)
print(f"\n\nChatbot: {summary}")