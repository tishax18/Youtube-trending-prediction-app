import streamlit as st
import joblib
from googleapiclient.discovery import build
from utils.predict_utils import predict_video_virality

# Load API Key
API_KEY = st.secrets["API_KEY"]
youtube = build("youtube", "v3", developerKey=API_KEY)

# Load Models
clf = joblib.load("clf.pkl")
xgb = joblib.load("xgb_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
stopwords = joblib.load("stopwords.pkl")

# UI
st.title("🎥 YouTube Trending Prediction App")
st.write("Paste a YouTube link to check if it can reach the trending section.")

url = st.text_input("Enter YouTube Video Link")

if st.button("Predict"):
    try:
        details, sentiment, pred_engage, pred_title, final, top_comments = \
            predict_video_virality(url, youtube, clf, xgb, vectorizer, stopwords)

        st.subheader("📺 Video Preview")
        st.video(f"https://www.youtube.com/watch?v={details['video_id']}")

        st.subheader("📌 Video Title")
        st.write(details["title"])

        st.subheader("📊 Stats")
        st.write(f"Views: {details['views']}")
        st.write(f"Likes: {details['likes']}")
        st.write(f"Comments: {details['comments']}")

        st.subheader("🧠 Model Predictions")
        st.write(f"Sentiment Score: {sentiment:.3f}")
        st.write(f"Engagement Model: {pred_engage}")
        st.write(f"Title Model: {pred_title}")

        st.subheader("🔮 Final Recommendation")
        st.success(final)

        st.subheader("💬 Top 5 Comments")
        for i, c in enumerate(top_comments, 1):
            st.write(f"**{i}.** {c}")

    except Exception as e:
        st.error(f"Error: {str(e)}")
