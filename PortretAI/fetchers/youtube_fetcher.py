import googleapiclient.discovery
import pandas as pd
from keys.fetcher_keys import YOUTUBE_DEVELOPER_KEY

class YouTubeCommentsFetcher:
    def __init__(self, api_key):
        self.api_service_name = "youtube"
        self.api_version = "v3"
        self.developer_key = api_key
        self.youtube = googleapiclient.discovery.build(self.api_service_name, self.api_version, developerKey=self.developer_key)
        self.comments = []

    def get_top_level_comments(self, video_id):
        nextPageToken = None
        while True:
            request = self.youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=nextPageToken
            )
            response = request.execute()
            
            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']
                self.comments.append([
                    comment['authorDisplayName'],
                    comment['publishedAt'],
                    comment['updatedAt'],
                    comment['likeCount'],
                    comment['textDisplay'],
                    'Top Level',
                    None
                ])
                self.get_replies(item['snippet']['topLevelComment']['id'])
            
            nextPageToken = response.get('nextPageToken')
            if not nextPageToken:
                break

    def get_replies(self, parent_id):
        nextPageToken = None
        while True:
            reply_request = self.youtube.comments().list(
                part="snippet",
                parentId=parent_id,
                maxResults=100,
                pageToken=nextPageToken
            )
            reply_response = reply_request.execute()
            
            for item in reply_response['items']:
                reply = item['snippet']
                self.comments.append([
                    reply['authorDisplayName'],
                    reply['publishedAt'],
                    reply['updatedAt'],
                    reply['likeCount'],
                    reply['textDisplay'],
                    'Reply',
                    parent_id
                ])
            
            nextPageToken = reply_response.get('nextPageToken')
            if not nextPageToken:
                break

    def fetch_comments(self, video_id):
        self.get_top_level_comments(video_id)
        df = pd.DataFrame(self.comments, columns=['author', 'published_at', 'updated_at', 'like_count', 'text', 'comment_type', 'parent_id'])
        return df