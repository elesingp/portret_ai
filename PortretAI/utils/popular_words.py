import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def find_important_words(comments, N=10):
    """
    Находит и возвращает N самых важных слов в комментариях.
    
    :param comments: Список предобработанных комментариев.
    :param N: Количество самых важных слов для вывода.
    :return: Список кортежей в формате (слово, сумма TF-IDF), отсортированный по убыванию важности.
    """
    # Векторизация комментариев
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(comments)
    
    # Суммирование значений TF-IDF по каждому слову
    word_sums = np.array(X.sum(axis=0)).flatten()
    
    # Получаем соответствие между индексами и словами
    words = vectorizer.get_feature_names_out()
    
    # Сортировка слов по убыванию их суммы TF-IDF
    sorted_words = sorted(list(zip(words, word_sums)), key=lambda x: x[1], reverse=True)
    
    # Возвращаем топ-N слов
    return sorted_words[:N]
