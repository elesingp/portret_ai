import requests
import time
from prompt_library.system_prompt_for_summarization import system_text_clustering
from keys.llm_keys import IAM_TOKEN, FOLDER_ID

def summarize_clusters_yandex_gpt(clusters):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {IAM_TOKEN}",
        "x-folder-id": FOLDER_ID
    }
    
    summaries = {}

    for cluster_id, texts in clusters.items():

        time.sleep(2)  # Приостанавливаем выполнение на 5 секунд

        cluster_text = " ".join(texts[:100])  # Ограничение количества текстов
        
        data = {
            "modelUri": f"gpt://{FOLDER_ID}/yandexgpt-lite",
            "completionOptions": {
                "stream": False,
                "temperature": 0.6,
                "maxTokens": 50
            },
            "messages": [
                {
                    "role": "system",
                    "text": system_text_clustering
                },
                {
                    "role": "user",
                    "text": cluster_text
                }
            ]
        }

        response = requests.post(
            "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            summary = response.json().get('result', {}).get('alternatives', [{}])[0].get('message', {}).get('text', 'Summary not available.')
            summaries[cluster_id] = summary
        else:
            summaries[cluster_id] = "Ошибка при генерации суммаризации: " + response.text

    return summaries


