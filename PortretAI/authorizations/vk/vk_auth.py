import requests

# Значения для запроса
user_id = '51859469'
version = '5.131'
access_token = '3df3ca453df3ca453df3ca45863ee49a4833df33df3ca455821a05b3f508e69b5a19e86'

# Формирование URL без access_token в query
url = f'https://api.vk.com/method/status.get?user_id={user_id}&v={version}'

# Формирование заголовков
headers = {
    'Authorization': f'Bearer {access_token}'
}

# Выполнение GET запроса с заголовками
response = requests.get(url, headers=headers)
print(response.json())
