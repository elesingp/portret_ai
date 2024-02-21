import os

from video_summarization.video_summarization_gpt import YoutubeVideoSummaryDescriptor
from keys.llm_keys import OPENAI_API_KEY
from paths.summary_flie_path import summary_file_path

video_ids = ["XsEhEm5l7Jk", "g5zubixGM8A"]  # Список Video ID для обработки
descriptor = YoutubeVideoSummaryDescriptor(OPENAI_API_KEY)

# Удаление файла с саммаризациями, если он существует, для начала работы с чистого листа
if os.path.exists(summary_file_path):
    os.remove(summary_file_path)

for video_id in video_ids:
    try:
        transcript, language_code = descriptor.get_transcript(video_id)
        summary = descriptor.summarize_gpt(transcript)
        descriptor.save_summary_to_file(video_id, language_code, summary, summary_file_path)
        print(f"Processed video ID: {video_id}")
    except Exception as e:
        print(f"Error processing video ID {video_id}: {e}")