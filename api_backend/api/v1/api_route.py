import itertools
import json

from typing import List

import numpy as np
import pandas as pd

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from gensim.models import Word2Vec

from services.metrics import (
    clean_text,
    lemm_text,
    calculate_russian_complexity,
    words_list,
    vectorize_text
)
from services.model_history import (
    load_model_history,
    save_model_history,
    save_model,
    load_model as load_model_from_storage,
    delete_model
)

from serializers.serializers import ModelConfig, ModelListResponse, RemoveResponse, FitResponse, PredictRequest,\
    PredictionResponse


router = APIRouter()


def prepare_pipeline(model_type: str, parsed_config: ModelConfig) -> Pipeline:
    """
    Подготавливает пайплайн модели.

    Args:
        model_type (str): Тип модели.
        parsed_config (ModelConfig): Конфигурация модели.

    Returns:
        Pipeline: Пайплайн для выбранной модели.
    """
    if model_type == "logistic":
        return Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(**parsed_config.hyperparameters))
        ])
    raise HTTPException(status_code=400, detail=f"Unsupported model type: {model_type}")


def process_texts(X: list, y: list) -> tuple:
    """
    Обрабатывает текстовые данные: очищает, лемматизирует и токенизирует.

    Args:
        X (list): Тексты (списки строк).
        y (list): Целевая переменная.

    Returns:
        tuple: Токенизированные и очищенные тексты.
    """
    author_list = list(itertools.chain(*[author.split() for author in pd.Series(y).str.lower().unique()]))
    texts_cleaned = [clean_text(str(text)) for text in X]
    texts_lemmatized = [lemm_text(text, author_list) for text in texts_cleaned]
    texts_tokenized = [words_list(text) for text in texts_lemmatized]
    return texts_tokenized, texts_cleaned


def prepare_features(texts_tokenized: list, texts_cleaned: list) -> tuple:
    """
    Создает признаки для модели.

    Args:
        texts_tokenized (list): Токенизированные тексты.
        texts_cleaned (list): Очищенные тексты.

    Returns:
        tuple: Признаки из токенов и числовые характеристики.
    """
    complexity_data = [calculate_russian_complexity(text) for text in texts_cleaned]
    complexity_df = pd.DataFrame(complexity_data)
    X_words = texts_tokenized
    X_heuristics = complexity_df[['num_words', 'avg_sentence_length', 'avg_word_length']]
    return X_words, X_heuristics


@router.post("/fit_csv", response_model=List[FitResponse])
async def fit_csv(
    file: UploadFile = File(...),
    target_column: str = Form(...),
    config: str = Form(...)
):
    """
    Обучение модели на основе CSV файла.

    Args:
        - file (UploadFile): CSV файл с данными, который отправляется в запросе.
        - target_column (str): Название целевой переменной (в нашем случае это всегда 'target').
        - config (str): JSON строка с конфигурацией модели.

    Returns:
        - List[FitResponse]:
            - message (str): Сообщение о завершении обучения и сохранения модели.
    """

    try:
        # Чтение поданного CSV файла из запроса
        df = pd.read_csv(file.file)

        # Проверка наличия таргета для обучения
        if not target_column or target_column not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid or missing 'target_column'. Available columns: {list(df.columns)}"
            )

        # Разделение данных на X и y
        X = df.drop(columns=[target_column]).astype(str).values.tolist()
        y = df[target_column].astype(str).values.tolist()

        # Парсинг конфигураций модели
        parsed_config = ModelConfig(**json.loads(config))
        model_type = parsed_config.ml_model_type
        model_id = parsed_config.id

        # Подготовка модели
        pipeline = prepare_pipeline(model_type, parsed_config)

    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to process CSV file: {str(e)}") from e

    try:

        texts_tokenized, texts_cleaned = process_texts(X, y)
        X_words, X_heuristics = prepare_features(texts_tokenized, texts_cleaned)

        # Разделение данных на тренировочные и тестовые
        X_heur_train, X_heur_test, X_words_train, X_words_test, y_train, y_test = train_test_split(
            X_heuristics, X_words, y, test_size=0.2, random_state=42
        )

        # Построение Word2Vec модели
        w2v_model = Word2Vec(sentences=X_words_train, vector_size=300, window=5, min_count=1, workers=4)
        save_model(f"{model_id}_w2v", w2v_model)

        # Преобразование текстовых данных в векторы
        X_train_vect = np.array([vectorize_text(text, w2v_model) for text in X_words_train])
        X_test_vect = np.array([vectorize_text(text, w2v_model) for text in X_words_test])

        # Объединение числовых и текстовых признаков
        X_train_combined = np.hstack([X_heur_train, X_train_vect])
        X_test_combined = np.hstack([X_heur_test, X_test_vect])

        # Обучение модели
        pipeline.fit(X_train_combined, y_train)

        # Предсказания
        y_pred = pipeline.predict(X_test_combined)

        # Вычисление метрик
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        conf_matrix = confusion_matrix(y_test, y_pred).tolist()

        # Сохранение модели в хранилище
        save_model(model_id, pipeline)

        # Обновление истории в регистре моделей
        history = load_model_history()
        history.append({
            "id": model_id,
            "type": model_type,
            "metrics": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "confusion_matrix": conf_matrix
            }
        })
        save_model_history(history)

        return [FitResponse(message=f"Model '{model_id}' trained and saved")]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}") from e


@router.post("/predict_request", response_model=List[PredictionResponse])
async def predict_request(requests: List[PredictRequest]):
    """
    Создание предсказаний для данных, переданных в JSON формате.

    Args:
        - requests (List[PredictRequest]): Список объектов `PredictRequest`, каждый из которых содержит:
            - X (List[str]): Список текстовых данных для предсказания.
            - id (str): Идентификатор модели, которая используется для предсказания.

    Returns:
        - List[PredictionResponse]: Список объектов `PredictionResponse`, каждый из которых содержит:
            - predictions (List[str]): Список предсказанных значений для соответствующих данных.
    """

    responses = []

    for request in requests:
        X = request.X
        model_id = request.id
        try:
            # Загрузка модели из хранилища
            model = load_model_from_storage(model_id)
            model_w2v = load_model_from_storage(f"{model_id}_w2v")
            author_list = []

            texts_cleaned = [clean_text(str(text)) for text in X]
            texts_lemmatized = [lemm_text(text, author_list) for text in texts_cleaned]
            texts_tokenized = [words_list(text) for text in texts_lemmatized]

            # Вычисление характеристик текста
            complexity_data = [calculate_russian_complexity(text) for text in texts_cleaned]
            complexity_df = pd.DataFrame(complexity_data)

            X_words = texts_tokenized

            X_heuristics = complexity_df[['num_words', 'avg_sentence_length', 'avg_word_length']]

            X_vect = np.array([vectorize_text(text, model_w2v) for text in X_words])

            # Объединение числовых и текстовых признаков
            X_combined = np.hstack([X_heuristics.values, X_vect])

            # Получаем предсказание
            predictions = model.predict(X_combined).tolist()
            responses.append(PredictionResponse(predictions=predictions))

        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found") from exc
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}") from e
    return responses


@router.get("/list_models", response_model=List[ModelListResponse])
async def list_models():
    """
    Возвращает список всех моделей с их метриками.

    Returns:
        - List[ModelListResponse]: Список объектов `ModelListResponse`, каждый из которых содержит:
            - models (List[ModelMetadata]): Список всех моделей с их метаданными, включая:
                - id (str): Идентификатор модели.
                - type (str): Тип модели (в нашем случае пока что только "logistic").
                - metrics (Dict[str, Union[float, List[List[int]]]]): Метрики модели.
    """

    history = load_model_history()
    return [
        {
            "models": history
        }
    ]


@router.delete("/remove/{model_id}", response_model=List[RemoveResponse])
async def remove(model_id: str):
    """
    Удаление модели по ее идентификатору из хранилища и истории.

    Args:
        - model_id (str): Идентификатор модели, которую необходимо удалить.

    Returns:
        - List[RemoveResponse]: Список с объектом `RemoveResponse`, который содержит:
            - message (str): Сообщение, подтверждающее успешное удаление модели из хранилища и истории.
    """

    try:
        delete_model(model_id)
        delete_model(f"{model_id}_w2v")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found in storage") from exc

    history = load_model_history()
    history = [entry for entry in history if entry["id"] != model_id]
    save_model_history(history)

    return [RemoveResponse(message=f"Model '{model_id}' removed")]
