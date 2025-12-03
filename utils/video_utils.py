import re
from googleapiclient.discovery import build

def extract_video_id(url):
    regex_list = [
        r"(?:https?://)?(?:www\.)?youtu\.be/([a-zA-Z0-9_-]{11})",
        r"(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})",
        r"v=([a-zA-Z0-9_-]{11})",
        r"embed/([a-zA-Z0-9_-]{11})",
        r"shorts/([a-zA-Z0-9_-]{11})"
    ]
    for regex in regex_list:
        match = re.search(regex, url)
        if match:
            return match.group(1)
    raise ValueError("Invalid YouTube link format.")



def get_video_details(video_url, youtube):
    video_id = extract_video_id(video_url)

    response = youtube.videos().list(
        part="snippet,statistics",
        id=video_id
    ).execute()

    if not response["items"]:
        raise ValueError("Video not found")

    item = response["items"][0]
    snippet = item["snippet"]
    stats = item["statistics"]

    details = {
        "video_id": video_id,
        "title": snippet.get("title", ""),
        "publish_time": snippet.get("publishedAt", ""),
        "channelTitle": snippet.get("channelTitle", ""),
        "thumbnail": snippet["thumbnails"]["high"]["url"],
        "tags": snippet.get("tags", []),
        "views": int(stats.get("viewCount", 0)),
        "likes": int(stats.get("likeCount", 0)),
        "comments": int(stats.get("commentCount", 0)),
        "description": snippet.get("description", "")
    }

    return details


