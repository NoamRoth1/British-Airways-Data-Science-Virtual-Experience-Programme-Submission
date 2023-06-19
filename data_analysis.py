import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt

from nltk.sentiment import SentimentIntensityAnalyzer

base_url = "https://www.airlinequality.com/airline-reviews/british-airways"
pages = 10
page_size = 100

reviews = []

# for i in range(1, pages + 1):
for i in range(1, pages + 1):

    print(f"Scraping page {i}")

    # Create URL to collect links from paginated data
    url = f"{base_url}/page/{i}/?sortby=post_date%3ADesc&pagesize={page_size}"

    # Collect HTML data from this page
    response = requests.get(url)

    # Parse content
    content = response.content
    parsed_content = BeautifulSoup(content, 'html.parser')
    for para in parsed_content.find_all("div", {"class": "text_content"}):
        reviews.append(para.get_text())

    print(f"   ---> {len(reviews)} total reviews")

df = pd.DataFrame()
df["reviews"] = reviews
df.head()
df.to_csv("data/BA_reviews.csv")

df['reviews'] = df['reviews'].str.replace('✅ Trip Verified', '')
df['reviews'] = df['reviews'].str.replace('❎', '')
df['reviews'] = df['reviews'].str.replace('Not Verified', '')
df['reviews'] = df['reviews'].str.replace('|', '')

# Extract a specific column as a Python list
reviews = df['reviews'].tolist()

# Initialize the SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Perform sentiment analysis on each review
sentiments = []
for review in reviews:
    sentiment_score = sia.polarity_scores(review)
    sentiment_label = 'Positive' if sentiment_score['compound'] > 0 else 'Negative'
    sentiments.append(sentiment_label)

# Create a DataFrame to store the reviews and sentiments
df = pd.DataFrame({'Review': reviews, 'Sentiment': sentiments})

# Count the number of positive and negative sentiments
sentiment_counts = df['Sentiment'].value_counts()

# Create a bar chart
plt.bar(sentiment_counts.index, sentiment_counts.values)
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Sentiment Distribution of Flight Reviews')
plt.savefig('BA bar chart.png')

df.to_csv("data/BA_reviews.csv", index=False)
