from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from api_backend.services.model_history import (
    load_model_history,
    save_model_history,
    save_model,
    load_model as load_model_from_storage,
    delete_model
)
from sklearn.linear_model import LogisticRegression
from api_backend.serializers.serializers import *

models = {}
router = APIRouter()

@router.post("/fit_request", response_model=List[FitResponse])
async def fit_request(request: FitRequest):
    """
    Обучение модели на основе JSON-запроса.
    """
    model_type = request.config.ml_model_type
    model_id = request.config.id

    # Инициализация модели
    if model_type == "logistic":
        model = LogisticRegression(**request.config.hyperparameters)
    else:
        raise HTTPException(
            status_code=400, detail=f"Unsupported model type: {model_type}"
        )

    try:
        # Обучение модели
        model.fit(request.X, request.y)
        save_model(model_id, model)

        # Обновление истории
        history = load_model_history()
        history.append({"id": model_id, "type": model_type})
        save_model_history(history)

        return [FitResponse(message=f"Model '{model_id}' trained and saved")]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.post("/fit_csv", response_model=List[FitResponse])
async def fit_csv(
    file: UploadFile = File(...),
    target_column: str = Form(...)
):
    """
    Обучение модели на основе CSV файла. `target_column` указывает целевую переменную.
    """
    try:
        # Читаем CSV файл
        df = pd.read_csv(file.file)

        # Проверяем наличие target_column
        if not target_column or target_column not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid or missing 'target_column'. Available columns: {list(df.columns)}"
            )

        # Выделяем X и y
        X = df.drop(columns=[target_column]).values.tolist()  # Все колонки, кроме целевой
        y = df[target_column].values.tolist()  # Целевая переменная

        # Конфигурация модели
        config = ModelConfig(
            id="model_from_csv",
            ml_model_type="logistic"
        )
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to process CSV file: {str(e)}"
        )

    model_type = config.ml_model_type
    model_id = config.id

    # Инициализация модели
    if model_type == "logistic":
        model = LogisticRegression(**config.hyperparameters)
    else:
        raise HTTPException(
            status_code=400, detail=f"Unsupported model type: {model_type}"
        )

    try:
        # Обучение модели
        model.fit(X, y)
        save_model(model_id, model)

        # Обновление истории
        history = load_model_history()
        history.append({"id": model_id, "type": model_type})
        save_model_history(history)

        return [FitResponse(message=f"Model '{model_id}' trained and saved")]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.post("/predict", response_model=List[PredictionResponse])
async def predict(requests: List[PredictRequest], file: UploadFile = File(None)):
    """
    Создание предсказаний. В момент вызова подгружается нужная модель по `model_id`.
    Принимает данные в формате List или CSV файл.
    """
    responses = []
    for request in requests:
        model_id = request.id
        try:
            # Если модель не загружена в память, подгружаем из файловой системы
            if model_id not in models:
                models[model_id] = load_model_from_storage(model_id)

            model = models[model_id]

            # Если пришел файл, то читаем его как CSV
            if file:
                df = pd.read_csv(file.file)
                X = df.values.tolist()  # Все колонки — признаки
            else:
                X = request.X  # Используем данные, пришедшие в запросе

            # Предсказания
            predictions = model.predict(X).tolist()
            responses.append(PredictionResponse(predictions=predictions))
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

    return responses


@router.get("/list_models", response_model=List[ModelListResponse])
async def list_models():
    """
    Возвращает список всех моделей (активных и заархивированных).
    """
    active_models = [
        ModelMetadata(id=model_id, type=model.__class__.__name__.lower())
        for model_id, model in models.items()
    ]
    history = load_model_history()
    archived_models = [
        ModelMetadata(id=entry["id"], type=entry["type"])
        for entry in history
        if entry["id"] not in models
    ]

    return [ModelListResponse(models=active_models + archived_models)]


@router.delete("/remove/{model_id}", response_model=List[RemoveResponse])
async def remove(model_id: str):
    """
    Удаление модели по ее идентификатору из памяти, файловой системы и истории обученных моделей.
    """
    if model_id in models:
        del models[model_id]

    try:
        delete_model(model_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found in storage")

    # Удаление из истории
    history = load_model_history()
    history = [entry for entry in history if entry["id"] != model_id]
    save_model_history(history)

    return [RemoveResponse(message=f"Model '{model_id}' removed")]


@router.delete("/remove_all", response_model=List[RemoveResponse])
async def remove_all():
    """
    Удаление всех моделей из памяти, файловой системы и истории обученных моделей.
    """
    removed_models = []

    # Удаляем из памяти
    for model_id in list(models.keys()):
        del models[model_id]

    # Удаляем из файловой системы
    history = load_model_history()
    for entry in history:
        model_id = entry["id"]
        try:
            delete_model(model_id)
            removed_models.append(RemoveResponse(message=f"Model '{model_id}' removed"))
        except FileNotFoundError:
            continue

    # Очищаем историю
    save_model_history([])

    if not removed_models:
        raise HTTPException(status_code=404, detail="No models found to remove")

    return removed_models