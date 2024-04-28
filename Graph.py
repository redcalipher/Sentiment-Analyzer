import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
df = pd.read_csv("C:/Users/Hendrixx/Downloads/FilteredGraphData.csv")

# Perform sentiment analysis on each tweet and store the sentiment in a new column
df['sentiment'] = df['tweets'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Categorize sentiment into positive, neutral, and negative
def categorize_sentiment(score):
    if score > 0:
        return 'positive'
    elif score == 0:
        return 'neutral'
    else:
        return 'negative'

df['sentiment_category'] = df['sentiment'].apply(categorize_sentiment)

# Count the number of positive, neutral, and negative tweets
sentiment_counts = df['sentiment_category'].value_counts()

# Plot the graph
plt.bar(sentiment_counts.index, sentiment_counts.values)
plt.title('Sentiment Analysis')
plt.xlabel('Sentiment')
plt.ylabel('Number of Tweets')
plt.xticks(rotation=45)
plt.show()
