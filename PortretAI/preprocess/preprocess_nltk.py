import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re

# Инициализация стеммера и загрузка стоп-слов для русского языка
stemmer = SnowballStemmer("russian")
russian_stopwords = stopwords.words("russian")

def preprocess_text_nltk(text):
    # Удаление специальных символов
    text = re.sub(r'\W', ' ', text)
    # Приведение к нижнему регистру
    text = text.lower()
    # Токенизация по пробелам
    words = text.split()
    # Удаление стоп-слов и стемминг
    filtered_words = [stemmer.stem(word) for word in words if word not in russian_stopwords]
    # Склеивание слов обратно в строку
    text = " ".join(filtered_words)
    return text

