import requests
import pandas as pd

class VKCommentsFetcher:
    def __init__(self, access_token):
        self.access_token = access_token
        self.api_version = '5.131'
        self.comments = []

    def get_post_comments(self, owner_id, post_id):
        """Получить комментарии к посту."""
        nextPageToken = '0'
        while nextPageToken is not None:
            response = requests.get(
                'https://api.vk.com/method/wall.getComments',
                params={
                    'access_token': self.access_token,
                    'v': self.api_version,
                    'owner_id': owner_id,
                    'post_id': post_id,
                    'count': 100,
                    'offset': nextPageToken
                }
            ).json()
            
            items = response.get('response', {}).get('items', [])
            for item in items:
                self.comments.append([
                    item.get('from_id'),  # В VK API используется from_id вместо authorDisplayName
                    item.get('date'),     # Время в формате Unixtime
                    '',                    # VK API не предоставляет updatedAt напрямую
                    item.get('likes', {}).get('count', 0),  # Количество лайков
                    item.get('text'),
                    'Top Level',
                    None  # parent_id для комментария верхнего уровня
                ])
            
            nextPageToken = response.get('response', {}).get('next_from', None)

    def fetch_comments(self, owner_id, post_id):
        self.get_post_comments(owner_id, post_id)
        df = pd.DataFrame(self.comments, columns=['author_id', 'date', 'updated_at', 'like_count', 'text', 'comment_type', 'parent_id'])
        return df

# Пример использования
access_token = 'ВАШ_ACCESS_TOKEN'
owner_id = '-1'  # Для группы используйте отрицательный ID
post_id = 'ID_ПОСТА'

vk_comments_fetcher = VKCommentsFetcher(access_token)
df_comments = vk_comments_fetcher.fetch_comments(owner_id, post_id)
print(df_comments)

