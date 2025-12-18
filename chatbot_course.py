import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from transformers import pipeline
import spacy

text = input("You: ")
text.lower()
text.translate(str.maketrans("", "", string.punctuation))
tokens = word_tokenize(text)
taggend_token = nltk.pos_tag(tokens)

nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
for ent_ in doc.ents:
    print(ent_.text, ent_.label)

sentiment_analyser = pipeline("sentiment-analysis")
result = sentiment_analyser(text)
response = ""

if result[0]['label'] == 'NEGATIVE':
    response += "I'm sorry for that, "

summarizer = pipeline("summarization")
summary = summarizer(text, max_length=100, min_length = 50, do_sample=False)
response += " "
response = str(response) + str(summary)
print(f"Chatbot: {response}")