# Стандартные библиотеки
import csv
import logging
import pandas as pd
from pathlib import Path

# Пути
from paths.comments_file_path import comments_file_path
from paths.summary_flie_path import summary_file_path

# Сторонние библиотеки
from flask import Flask, render_template, request, flash
from flask_bootstrap import Bootstrap

# Модули приложения
from fetchers.youtube_fetcher import YouTubeCommentsFetcher
from video_summarization.video_summarization_gpt import YoutubeVideoSummaryDescriptor
from keys.fetcher_keys import YOUTUBE_DEVELOPER_KEY
from keys.llm_keys import OPENAI_API_KEY
from prompt_library.system_prompt_for_summarization import system_text_clustering

# Главный класс
from baseline.results import TextClusterAnalysis

"""
YouTube Comments Analyzer

Это Flask приложение предназначено для анализа комментариев из видео на YouTube.
Пользователи могут ввести URL видео на YouTube, и приложение загрузит комментарии,
проанализирует их и отобразит результаты анализа.

Features:
- Загрузка комментариев с YouTube по URL видео.
- Анализ текста комментариев и вывод результатов.
- Отображение системного текста и анализа в красивом фронтенде.

Requirements:
- Flask и Flask-Bootstrap для бэкенда и фронтенда.
- Собственные модули для загрузки комментариев и анализа текста.
"""

app = Flask(__name__)
Bootstrap(app)

# Необходимо добавить секретный ключ для защиты сессий, где хранятся flash сообщения
app.secret_key = '1234'

@app.route('/')
def home():
    # Страница с описанием проекта и поиском URL
    return render_template('search_bar.html')

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    video_urls = []
    form_submitted = False
    combined_df_comments = pd.DataFrame()  # Initialize an empty DataFrame

    if request.method == 'POST':
        form_submitted = True
        # Получаем URL-адреса, разделённые запятыми
        video_urls_input = request.form.get('video_urls', '').strip()  # Используем 'video_urls'

        if Path(comments_file_path).exists():
            Path(comments_file_path).unlink()

        if Path(summary_file_path).exists():
            Path(summary_file_path).unlink()

        if video_urls_input:
            # Разделяем ввод по запятым и удаляем пробельные символы вокруг каждого URL
            video_urls = [url.strip() for url in video_urls_input.split(',')]
            for video_url in video_urls:
                try:
                    # Извлечение комментариев
                    video_id = video_url.split('v=')[-1].split('&')[0]
                    fetcher = YouTubeCommentsFetcher(YOUTUBE_DEVELOPER_KEY)
                    df_comments = fetcher.fetch_comments(video_id)

                    if not df_comments.empty:
                        combined_df_comments = pd.concat([combined_df_comments, df_comments], ignore_index=True)
                    else:
                        flash(f'Не удалось загрузить комментарии для {video_url} или комментариев нет.', 'warning')

                    # Транскрибация видео
                    descriptor = YoutubeVideoSummaryDescriptor(OPENAI_API_KEY)
                    transcript, language_code = descriptor.get_transcript(video_id)
                    summary = descriptor.summarize_gpt(transcript)
                    descriptor.save_summary_to_file(video_id, language_code, summary, summary_file_path)
                    print(f"Processed video ID: {video_id}")

                except Exception as e:
                    flash(f'Ошибка при загрузке содержимого для {video_url}: {e}', 'error')    

            if not combined_df_comments.empty:
                combined_df_comments.to_csv(comments_file_path, index=False, quoting=csv.QUOTE_ALL)
                flash('Комментарии успешно загружены и объединены.', 'success')
            else:
                flash('Не удалось загрузить комментарии для предоставленных URL.', 'warning')
        else:
            flash('Пожалуйста, введите один или несколько URL через запятую.', 'warning')

    results = None
    if form_submitted and not combined_df_comments.empty:
        analysis = TextClusterAnalysis(comments_file_path)  # Specify the path to the combined CSV
        results = analysis.run_analysis()
        
    return render_template('index.html', results=results, system_text=system_text_clustering, video_url=video_urls)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(debug=True)