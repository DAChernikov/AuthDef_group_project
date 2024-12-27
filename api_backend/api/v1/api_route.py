from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from serializers.serializers import *
from services.model_history import (
    load_model_history,
    save_model_history,
    save_model,
    load_model as load_model_from_storage,
    delete_model
)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix
)

import json


router = APIRouter()


@router.post("/fit_request", response_model=List[FitResponse])
async def fit_request(request: FitRequest):
    """
    Обучение модели на основе JSON-запроса.
    """
    model_type = request.config.ml_model_type
    model_id = request.config.id

    if model_type == "logistic":
        model = LogisticRegression(**request.config.hyperparameters)
    else:
        raise HTTPException(
            status_code=400, detail=f"Unsupported model type: {model_type}"
        )

    try:
        # Подготовка обучающего датасета
        X, y = request.X, request.y
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Обучение модели
        model.fit(X_train, y_train)

        # Предсказания
        y_pred = model.predict(X_test)

        # Вычисление метрик
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        conf_matrix = confusion_matrix(y_test, y_pred).tolist()

        # Сохранение модели в хранилище
        save_model(model_id, model)

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
        X = df.drop(columns=[target_column]).values.tolist()
        y = df[target_column].values.tolist()

        # Парсинг конфигураций модели
        parsed_config = ModelConfig(**json.loads(config))
        model_type = parsed_config.ml_model_type
        model_id = parsed_config.id

        # Подготовка модели
        if model_type == "logistic":
            model = LogisticRegression(**parsed_config.hyperparameters)
        else:
            raise HTTPException(
                status_code=400, detail=f"Unsupported model type: {model_type}"
            )
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to process CSV file: {str(e)}"
        )

    try:
        # Подготовка выборок для обучения
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Обучение модели
        model.fit(X_train, y_train)

        # Предсказания
        y_pred = model.predict(X_test)

        # Вычисление метрик
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        conf_matrix = confusion_matrix(y_test, y_pred).tolist()

        # Сохранение модели в хранилище
        save_model(model_id, model)

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


@router.post("/predict_request", response_model=List[PredictionResponse])
async def predict_request(requests: List[PredictRequest]):
    """
    Создание предсказаний для данных, переданных в JSON формате.
    """
    responses = []
    for request in requests:
        model_id = request.id
        try:
            # Загрузка модели из хранилища
            model = load_model_from_storage(model_id)
            X = request.X
            predictions = model.predict(X).tolist()
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
            removed_models.append(RemoveResponse(message=f"Model '{model_id}' removed"))
        except FileNotFoundError:
            continue

    # Очистка истории из регистра моделей
    save_model_history([])

    if not removed_models:
        raise HTTPException(status_code=404, detail="No models found to remove")

    return removed_models