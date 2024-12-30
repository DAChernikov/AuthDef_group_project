import re
import nltk
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize

nltk.download('punkt_tab')
import pymorphy3

morph = pymorphy3.MorphAnalyzer()

def clean_text(text: str) -> str:
    """
    Очистка текста от лишних символов и приведение к нижнему регистру.
    """
    text = re.sub(r'&#\d+;|&#x[0-9a-fA-F]+;', '', text)  # Удаление unicode-символов
    # print('ha')
    text = text.replace('\n', ' ')  # Замена переносов строк на пробел
    text = re.sub(r'[^А-Яа-я0-9\(\).,!?: \-]', '', text.lower())  # Удаление неподходящих символов
    text = re.sub(r'\s+', ' ', text).strip()  # Удаление лишних пробелов
    return text

def lemm_text(text: str, author_list: list) -> str:
    """
    Лемматизация текста с фильтрацией.
    """
    tokens = word_tokenize(text)
    lemmas = [morph.parse(token)[0].normal_form for token in tokens if token.isalnum()]
    final_lemmas = [
        word for word in lemmas
        if word not in author_list and len(word) > 2 and not re.fullmatch(r'\d+|[^A-Za-zА-Яа-яЁё]+', word)
    ]
    return ' '.join(final_lemmas)

def calculate_russian_complexity(text: str):
    """
    Вычисление характеристик сложности текста.
    """
    sentences = sent_tokenize(text, language="russian")
    words = word_tokenize(text, language="russian")
    words = [word.lower() for word in words if word.isalpha()]

    num_sentences = len(sentences)
    num_words = len(words)
    avg_sentence_length = num_words / num_sentences if num_sentences > 0 else 0
    avg_word_length = sum(len(word) for word in words) / num_words if num_words > 0 else 0

    return {
        "num_words": num_words,
        "avg_sentence_length": avg_sentence_length,
        "avg_word_length": avg_word_length,
    }

def words_list(text: str):
    words = re.findall(r'\b\w+\b', text)
    return [w.lower() for w in words]

def vectorize_text(text, model):
    vectors = [model.wv[word] for word in text if word in model.wv]
    if len(vectors) == 0:
        return np.zeros(model.vector_size)  # Если нет слов из текста, возвращаем нулевой вектор
    return np.mean(vectors, axis=0)
