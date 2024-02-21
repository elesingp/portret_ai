from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi

from prompt_library.system_prompt_for_video import system_prompt_for_video

class YoutubeVideoSummaryDescriptor():
    def __init__(self, openai_key):
        self.client = OpenAI(api_key=openai_key)

    def get_transcript(self, video_id):
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        try:
            transcript = transcript_list.find_manually_created_transcript()
            language_code = transcript.language_code  # Save the detected language
        except:
            # If no manual transcript is found, try fetching an auto-generated transcript in a supported language
            try:
                generated_transcripts = [trans for trans in transcript_list if trans.is_generated]
                transcript = generated_transcripts[0]
                language_code = transcript.language_code  # Save the detected language
            except:
                # If no auto-generated transcript is found, raise an exception
                raise Exception("No suitable transcript found.")

        full_transcript = " ".join([part['text'] for part in transcript.fetch()])

        # Обрезаем первые 50 слов
        trimmed_transcript = " ".join(full_transcript.split()[:50])

        return trimmed_transcript, language_code  # Return both the transcript and detected language

    def summarize_gpt(self, video_text):
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt_for_video
                },
                {
                    "role": "user",
                    "content": video_text
                }
            ],
            temperature=0.5,
            max_tokens=300,
            top_p=1
        )
        
        if response.choices:
            summary = response.choices[0].message.content  # Corrected access to message content
        else:
            summary = "Summary not available."

        return summary
    
    def save_summary_to_file(self, video_id, language_code, summary, file_path='summaries.txt'):
        with open(file_path, 'a', encoding='utf-8') as file:
            file.write(f"Video ID: {video_id}\nLanguage: {language_code}\nSummary: {summary}\n\n")
