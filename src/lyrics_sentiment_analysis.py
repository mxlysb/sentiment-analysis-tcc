import pandas as pd
import re

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

df = pd.read_csv("../Billboard_Hot100_Songs_with_Lyrics_Sample.csv")

stop_words = set(stopwords.words('english'))

def preprocess_lyrics(lyrics):
    if pd.isna(lyrics):
        return ""
    lyrics = re.sub(r'\[.*?\]', '', lyrics)
    lyrics = re.sub(r'\n', ' ', lyrics)
    lyrics = re.sub(r'[^a-zA-Z\s]', '', lyrics)
    lyrics = re.sub(r'ContributorTo Each His Own Lyrics', '', lyrics)
    lyrics = re.sub(r'contributors', '', lyrics)
    lyrics = re.sub(r'contributor', '', lyrics)
    lyrics = re.sub(r'lyrics', '', lyrics)
    lyrics = lyrics.lower()
    lyrics = ' '.join([word for word in lyrics.split() if word not in stop_words])
    lyrics = re.sub(r'\s+', ' ', lyrics).strip()

    return lyrics

df['Clean_lyrics'] = df['lyrics'].apply(preprocess_lyrics)
df = df[df['Clean_lyrics'] != ""]

analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment_vader(lyrics):
    sentiment = analyzer.polarity_scores(lyrics)
    return sentiment

def analyze_sentiment_textblob(lyrics):
    blob = TextBlob(lyrics)
    return blob.sentiment

df['sentiment_vader'] = df['Clean_lyrics'].apply(analyze_sentiment_vader)
df['sentiment_textblob'] = df['Clean_lyrics'].apply(analyze_sentiment_textblob)

df['vader_pos'] = df['sentiment_vader'].apply(lambda x: x['pos'])
df['vader_neu'] = df['sentiment_vader'].apply(lambda x: x['neu'])
df['vader_neg'] = df['sentiment_vader'].apply(lambda x: x['neg'])
df['vader_compound'] = df['sentiment_vader'].apply(lambda x: x['compound'])

df['textblob_polarity'] = df['sentiment_textblob'].apply(lambda x: x.polarity)
df['textblob_subjectivity'] = df['sentiment_textblob'].apply(lambda x: x.subjectivity)

print(df[['Song', 'sentiment_vader', 'sentiment_textblob']].head())

df.to_csv('../Billboard_Hot100_Songs_with_Sentiment.csv', index=False)

print("Processo concluído e CSV salvo como 'Billboard_Hot100_Songs_with_Sentiment.csv'")
