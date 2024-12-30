import itertools
import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from serializers.serializers import *
from sklearn.model_selection import train_test_split
import os
from services.model_history import (
    load_model_history,
    save_model_history,
    save_model,
    load_model as load_model_from_storage,
    delete_model
)
from gensim.models import Word2Vec
from services.metrics import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix
)

import json

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

router = APIRouter()


@router.post("/fit_request", response_model=List[FitResponse])
async def fit_request(request: FitRequest):
    """
    Обучение модели на основе JSON-запроса.
    """
    # параметры модели 
    model_type = request.config.ml_model_type
    model_id = request.config.id

    if model_type == "logistic":
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(**request.config.hyperparameters))
        ])
    else:
        raise HTTPException(
            status_code=400, detail=f"Unsupported model type: {model_type}"
        )

    try:
        # Список авторов (для удаления из текста)
        author_list = list(
            itertools.chain(*[author.split() for author in pd.Series(request.y).str.lower().unique()])
        )
        # Очистка и обработка текстов
        texts_cleaned = [clean_text(text) for text in request.X]
        texts_lemmatized = [lemm_text(text, author_list) for text in texts_cleaned]
        texts_tokenized = [words_list(text) for text in texts_lemmatized]

        # Вычисление характеристик сложности текста
        complexity_data = [calculate_russian_complexity(text) for text in texts_cleaned]
        complexity_df = pd.DataFrame(complexity_data)
        
        X_words = pd.DataFrame(texts_tokenized)
        
        # Создание выборок для обучения
        X_heuristics = complexity_df[['num_words', 'avg_sentence_length', 'avg_word_length']]
        y = request.y

        # Разделение данных на тренировочные и тестовые выборки
        X_heur_train, X_heur_test, X_words_train, X_words_test, y_train, y_test = train_test_split(
            X_heuristics, X_words, y, test_size=0.2, random_state=42
        )

        # Построение Word2Vec модели
        w2v_model = Word2Vec(sentences=X_words_train, vector_size=300, window=5, min_count=1, workers=4)

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
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@router.post("/fit_csv", response_model=List[FitResponse])
async def fit_csv(
    file: UploadFile = File(...),
    target_column: str = Form(...),
    config: str = Form(...)
):
    """
    Обучение модели на основе CSV файла. `target_column` указывает целевую переменную.
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

        # Разделение поданных данных на X и y
        X = df.drop(columns=[target_column]).astype(str).values.tolist()
        y = df[target_column].astype(str).values.tolist()

        # Парсинг конфигураций модели
        parsed_config = ModelConfig(**json.loads(config))
        model_type = parsed_config.ml_model_type
        model_id = parsed_config.id

        # Подготовка модели
        if model_type == "logistic":
            pipeline = Pipeline([
                ('scaler', StandardScaler()),  # Стандартизация данных
                ('classifier', LogisticRegression(**parsed_config.hyperparameters))  # Логистическая регрессия
            ])
        else:
            raise HTTPException(
                status_code=400, detail=f"Unsupported model type: {model_type}"
            )
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to process CSV file: {str(e)}"
        )

    try:
        # Список авторов (для удаления из текста)
        author_list = list(
            itertools.chain(*[author.split() for author in pd.Series(y).str.lower().unique()])
        )
        # Очистка и обработка текстов
        texts_cleaned = [clean_text(str(text)) for text in X]
        texts_lemmatized = [lemm_text(text, author_list) for text in texts_cleaned]
        texts_tokenized = [words_list(text) for text in texts_lemmatized]

        # Вычисление характеристик сложности текста
        complexity_data = [calculate_russian_complexity(text) for text in texts_cleaned]
        complexity_df = pd.DataFrame(complexity_data)

        X_words = texts_tokenized

        # Создание выборок для обучения
        X_heuristics = complexity_df[['num_words', 'avg_sentence_length', 'avg_word_length']]

        # Разделение данных на тренировочные и тестовые выборки
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
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.post("/save_model", response_model=List[SaveResponse])
async def post_save_model(
        file: UploadFile = File(...)
):
    """
    Сохранение готовой модели.
    """
    try:
        file_name, file_extension = os.path.splitext(file.filename)
        history = load_model_history()
        history.append({
            "id": file_name,
            "type": "pretrained",
            "metrics": {}
        })

        if not file_name.endswith("w2v"):
            save_model_history(history)

        save_model(file_name, file)

        return [SaveResponse(message=f"Model '{file_name}' saved")]
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"Saving failed: {str(e)}")


@router.post("/predict_request", response_model=List[PredictionResponse])
async def predict_request(requests: List[PredictRequest]):
    """
    Создание предсказаний для данных, переданных в JSON формате.
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

            # Вычисление характеристик сложности текста
            complexity_data = [calculate_russian_complexity(text) for text in texts_cleaned]
            complexity_df = pd.DataFrame(complexity_data)

            X_words = texts_tokenized

            # Создание выборок для обучения
            X_heuristics = complexity_df[['num_words', 'avg_sentence_length', 'avg_word_length']]

            X_vect = np.array([vectorize_text(text, model_w2v) for text in X_words])

            # Объединение числовых и текстовых признаков
            X_combined = np.hstack([X_heuristics.values, X_vect])

            # Получаем предсказания
            predictions = model.predict(X_combined).tolist()
            responses.append(PredictionResponse(predictions=predictions))

        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")
    return responses


@router.post("/predict_csv", response_model=List[PredictionResponse])
async def predict_csv(
    file: UploadFile = File(...),
    model_id: str = Form(...)
):
    """
    Предсказание на основе CSV файла. Используется модель с указанным `model_id`.
    """
    try:
        # Чтение поданного CSV файла
        df = pd.read_csv(file.file)
        if df.empty:
            raise HTTPException(status_code=400, detail="Provided CSV file is empty")

        # Загрузка модели из хранилища
        model = load_model_from_storage(model_id)

        # Предсказания
        X = df.values.tolist()
        predictions = model.predict(X).tolist()

        return [PredictionResponse(predictions=predictions)]
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("/list_models", response_model=List[ModelListResponse])
async def list_models():
    """
    Возвращает список всех моделей с их метриками.
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
    """
    try:
        delete_model(model_id)
        delete_model(f"{model_id}_w2v")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found in storage")

    # Удаление модели из регистра моделей
    history = load_model_history()
    history = [entry for entry in history if entry["id"] != model_id]
    save_model_history(history)

    return [RemoveResponse(message=f"Model '{model_id}' removed")]


@router.delete("/remove_all", response_model=List[RemoveResponse])
async def remove_all():
    """
    Удаление всех моделей из хранилища и истории.
    """
    removed_models = []

    # Удаление всех моделей из хранилища
    history = load_model_history()
    for entry in history:
        model_id = entry["id"]
        try:
            delete_model(model_id)
            delete_model(f"{model_id}_w2v")
            removed_models.append(RemoveResponse(message=f"Model '{model_id}' removed"))
        except FileNotFoundError:
            continue

    # Очистка истории из регистра моделей
    save_model_history([])

    if not removed_models:
        raise HTTPException(status_code=404, detail="No models found to remove")

    return removed_models
