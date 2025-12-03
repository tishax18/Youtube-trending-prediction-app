import numpy as np
from textblob import TextBlob

def get_top_comments(video_id, youtube, max_comments=5):
    comments = []

    response = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=max_comments,
        textFormat="plainText"
    ).execute()

    for item in response["items"]:
        text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        comments.append(text)

    return comments


def get_comment_sentiment(video_id, youtube, max_comments=30):
    sentiments = []

    response = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=max_comments,
        textFormat="plainText"
    ).execute()

    for item in response["items"]:
        text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        polarity = TextBlob(text).sentiment.polarity
        sentiments.append(polarity)

    return np.mean(sentiments) if sentiments else 0
