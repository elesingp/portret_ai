import pandas as pd

# Предобработка текста
from preprocess.preprocess_v1 import preprocess_text_morph

# Векторизация
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from vectorizers.w2v_vectorizer import w2v_vectorizer
# Кластеризация
from sklearn.cluster import KMeans

# Метрики
from metrics.silhouette_score import evaluate_clustering

# Дополнительные утилиты
from utils.popular_words import find_important_words
from collections import defaultdict

# Саммаризация кластеризированного текста
from cluster_summarization.summarization_yandex import summarize_clusters_yandex_gpt
from cluster_summarization.summarization_gpt import summarize_clusters_gpt

# Генерация репорта
from report_generation.report_generation_gpt import generate_overall_report

class TextClusterAnalysis:
    def __init__(self, comments_path):
        self.comments_path = comments_path
        self.comments = pd.read_csv(comments_path)
        self.vectorizer = w2v_vectorizer()
        self.k = 10  # Предполагаемое количество кластеров

    def preprocess_comments(self):
        return [preprocess_text_morph(comment) for comment in self.comments['text']]

    def vectorize_text(self, preprocessed_comments):
        X = self.vectorizer.vectorize(preprocessed_comments)
        return X

    def cluster_comments(self, X_normalized):
        model = KMeans(n_clusters=self.k, random_state=42)
        #print(X_normalized)
        model.fit(X_normalized)
        return model.labels_

    def analyze_clusters(self, labels, comments):
        clusters = defaultdict(list)
        #print(comments)
        for comment, label in zip(comments.text, labels):
            #print(label)
            clusters[label].append(comment)
        #print(clusters)
        return clusters

    def summarize_clusters(self, clusters):
        return summarize_clusters_gpt(clusters)

    def generate_report(self, cluster_summaries):
        return generate_overall_report(cluster_summaries)

    def run_analysis(self):
        preprocessed_comments = self.preprocess_comments()
        X_normalized = self.vectorize_text(preprocessed_comments)
        labels = self.cluster_comments(X_normalized)
        clusters = self.analyze_clusters(labels, self.comments)
        
        #important_words = find_important_words(preprocessed_comments, N=10)
        
        score = evaluate_clustering(X_normalized, labels)
        
        cluster_summaries = self.summarize_clusters(clusters)

        report = self.generate_report(cluster_summaries)
        
        # Модифицирован для возврата результатов вместо печати
        return {
            #'important_words': important_words,
            'score': score,
            'cluster_summaries': cluster_summaries,
            'report': report,
        }