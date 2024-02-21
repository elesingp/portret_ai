from openai import OpenAI
from keys.llm_keys import OPENAI_API_KEY
from prompt_library.system_prompt_for_summarization import system_text_clustering

def summarize_clusters_gpt(clusters):
    client = OpenAI(api_key=OPENAI_API_KEY)
    summaries = {}
    for cluster_id, comments in clusters.items():
        # Агрегируем комментарии кластера в один текст
        cluster_text = " ".join(comments[:100])  # Ограничиваем количество комментариев для суммаризации

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": system_text_clustering
                },
                {
                    "role": "user",
                    "content": cluster_text
                }
            ],
            temperature=0.5,
            max_tokens=600,
            top_p=1
        )
        
        # Правильный способ извлечения содержимого сообщения
        if response.choices:
            summary = response.choices[0].message.content  # Исправлено на правильный доступ к тексту сообщения
        else:
            summary = "Summary not available."
        summaries[cluster_id] = summary

    return summaries





