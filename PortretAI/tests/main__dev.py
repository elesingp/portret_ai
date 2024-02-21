import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# Главный класс
from baseline.results import TextClusterAnalysis
from paths.comments_file_path import comments_file_path

if __name__ == "__main__":
    
    # Создание экземпляра класса анализа
    analysis = TextClusterAnalysis(comments_file_path)
    
    # Выполнение анализа
    results = analysis.run_analysis()
    
    # Отображение важных слов
    #print("\nВажные слова:")
    #for word, importance in results['important_words']:
        #print(f"{word}: {importance}")
    
    # Отображение силуэтного коэффициента
    print(f"\nСилуэтный коэффициент: {results['score']}")
    
    # Отображение саммаризации кластеров
    print("\nСаммаризация кластеров:")
    for cluster_id, summary in results['cluster_summaries'].items():
        print(f"\nКластер {cluster_id}:")
        print(summary)

    print("\nОтчет")
    print(results['report'])
