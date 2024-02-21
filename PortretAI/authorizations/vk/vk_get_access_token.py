import requests

def exchange_silent_auth_token(api_version, silent_token, service_token, uuid):
    url = "https://api.vk.com/method/auth.exchangeSilentAuthToken"
    payload = {
        "v": api_version,
        "token": silent_token,
        "access_token": service_token,
        "uuid": uuid
    }
    response = requests.post(url, data=payload)
    return response.json()

# Пример использования
api_version = "5.131"
silent_token = "C0I5ueufDvIkNR2dBz6k"
service_token = "3df3ca453df3ca453df3ca45863ee49a4833df33df3ca455821a05b3f508e69b5a19e86"
uuid = "51859469"

access_token_info = exchange_silent_auth_token(api_version, silent_token, service_token, uuid)
print(access_token_info)
