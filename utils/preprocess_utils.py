import numpy as np
import pandas as pd
from datetime import datetime
import string
import re
from textblob import TextBlob
from nltk.stem import PorterStemmer

ps = PorterStemmer()

def preprocess_for_engagement(details):
    views = details["views"]
    likes = details["likes"]
    comments = details["comments"]
    tags = details["tags"]
    title = details["title"]
    publish_time = details["publish_time"]

    rs = np.random.RandomState(42)
    frac = rs.beta(2.0, 6.0, size=1)
    frac = np.clip(frac, 0.05, 0.6)[0]

    first24_views = int(views * frac)
    first24_likes = int(likes * frac)
    first24_comments = int(comments * frac)

    row = {
        "first24_views": first24_views,
        "first24_likes": first24_likes,
        "first24_comments": first24_comments,
        "like_view_ratio_24h": first24_likes / (first24_views + 1),
        "comment_view_ratio_24h": first24_comments / (first24_views + 1),
        "tag_count": len(tags),
        "title_length": len(str(title)),
        "publish_hour": datetime.fromisoformat(publish_time.replace("Z", "")).hour
                        if "Z" in publish_time else 0,
        "title_sentiment": TextBlob(str(title)).sentiment.polarity,
        "engagement_24h": first24_likes + first24_comments
    }

    return pd.DataFrame([row])


def preprocess_for_title_with_clf(details, vectorizer, stopwords):
    title = details["title"]

    def clean_text(text):
        text = "".join([w.lower() for w in text if w not in string.punctuation])
        tokens = re.split(r"\W+", text)
        text = [ps.stem(word) for word in tokens if word not in stopwords]
        return " ".join(text)

    cleaned_title = clean_text(title)
    return vectorizer.transform([cleaned_title])
