from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    Doc
)

# Инициализация инструментов Natasha
segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)

def preprocess_text_natasha(text):
    doc = Doc(text)
    
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('russian'))
    # Сегментация текста на токены
    doc.segment(segmenter)
    
    # Морфологический анализ и лемматизация
    doc.tag_morph(morph_tagger)
    
    for token in doc.tokens:
        token.lemmatize(morph_vocab)
    
    # Удаление стоп-слов и создание списка лемматизированных токенов
    lemmatized_tokens = [_.lemma for _ in doc.tokens if _.lemma not in stop_words]
    
    # Возврат предобработанного текста
    return " ".join(lemmatized_tokens)