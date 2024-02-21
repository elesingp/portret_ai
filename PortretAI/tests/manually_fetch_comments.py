from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import csv
import pandas as pd
from paths.comments_file_path import comments_file_path



from keys.fetcher_keys import YOUTUBE_DEVELOPER_KEY
from fetchers.youtube_fetcher import YouTubeCommentsFetcher

def fetch_and_combine_comments_manually(video_ids, api_key):
    fetcher = YouTubeCommentsFetcher(api_key)
    combined_df = pd.DataFrame()  # Инициализируем пустой DataFrame для сбора всех комментариев

    if Path(comments_file_path).exists():
        Path(comments_file_path).unlink()

    for video_id in video_ids:
        try:
            df = fetcher.fetch_comments(video_id)
            if not df.empty:
                combined_df = pd.concat([combined_df, df], ignore_index=True)
            else:
                print(f"No comments found for video ID: {video_id}")
        except Exception as e:
            print(f"Error fetching comments for video ID {video_id}: {e}")

    return combined_df

if __name__ == "__main__":
    api_key = YOUTUBE_DEVELOPER_KEY
    video_ids = ["aqcfE9xPZew", "LjBbzOiDp7k", "XsEhEm5l7Jk"]  # Пример списка video_id
    combined_comments_df = fetch_and_combine_comments_manually(video_ids, api_key)

    if not combined_comments_df.empty:
        combined_comments_df.to_csv(comments_file_path, index=False, quoting=csv.QUOTE_ALL)
        print(f"Combined comments saved to {comments_file_path}")
    else:
        print("No comments were fetched.")