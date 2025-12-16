from transformers import pipeline

sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)
print("Sentiment analysis model loaded successfully!")

review = input("Give me a review for our product: ")
result = sentiment_analyzer(review)


if result[0]['label'] == 'POSITIVE':
    print("Thank you for your review!")

else:
    print("We are so sorry about that!")