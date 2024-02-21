from openai import OpenAI
from keys.llm_keys import OPENAI_API_KEY
from prompt_library.system_prompt_for_report import system_prompt_report

def generate_overall_report(cluster_summaries):
    client = OpenAI(api_key=OPENAI_API_KEY)
    # Сначала создаем суммаризации для каждого кластера
    
    # Агрегируем суммаризации в один текстовый блок для общего отчета
    summaries_text = "\n\n".join([f"Кластер {cluster_id}: {summary}" for cluster_id, summary in cluster_summaries.items()])
    print(system_prompt_report)
    # Формируем запрос с агрегированным текстом и системным промптом для общего отчета
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": system_prompt_report
            },
            {
                "role": "user",
                "content": summaries_text
            }
        ],
        temperature=0.5,
        max_tokens=2000,  # Увеличиваем количество токенов для более детального отчета
        top_p=1
    )
    
    # Извлекаем и возвращаем сгенерированный отчет
    if response.choices:
        report = response.choices[0].message.content  # Исправлено на правильный доступ к тексту сообщения
    else:
        report = "Общий отчет недоступен."
    
    return report
