from sklearn.metrics import silhouette_score

def evaluate_clustering(X, labels):
    """
    Оценивает качество кластеризации, используя силуэтный коэффициент.
    
    :param X: Матрица признаков объектов после TF-IDF векторизации.
    :param labels: Массив меток кластеров, полученных после кластеризации.
    :return: Силуэтный коэффициент для данной кластеризации.
    """
    score = silhouette_score(X, labels, metric='euclidean')
    return score