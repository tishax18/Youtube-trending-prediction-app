from utils.video_utils import get_video_details
from utils.comment_utils import get_top_comments, get_comment_sentiment
from utils.preprocess_utils import preprocess_for_engagement, preprocess_for_title_with_clf

def predict_video_virality(video_url, youtube, clf, xgb, vectorizer, stopwords):
    details = get_video_details(video_url, youtube)
    video_id = details["video_id"]

    # Sentiment
    sentiment_score = get_comment_sentiment(video_id, youtube)
    top_comments = get_top_comments(video_id, youtube)

    # Engagement model
    X_engage = preprocess_for_engagement(details)
    pred_engage = xgb.predict(X_engage)[0]

    # Title model
    title_vec = preprocess_for_title_with_clf(details, vectorizer, stopwords)
    pred_title = clf.predict(title_vec)[0]

    # Final decision
    if sentiment_score > 0.15 or pred_engage == 1 or pred_title == 1:
        final = "🔥 RECOMMENDED for Trending Section"
    else:
        final = "❄️ NOT Recommended"

    if pred_engage == 1 or pred_title == 1:
        final = "🔥 RECOMMENDED for Trending Section"
    else:
        if sentiment_score > 0.40:
           final = "⚠️ Potentially Good, but Models Predict NOT Trending"
        else:
           final = "❄️ NOT Recommended"

    return details, sentiment_score, pred_engage, pred_title, final, top_comments
