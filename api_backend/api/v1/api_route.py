from fastapi import APIRouter, HTTPException, UploadFile, File
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

@router.post("/fit", response_model=List[FitResponse])
async def fit(requests: List[FitRequest], file: UploadFile = File(None)):
    """
    Обучение и сохранение модели на основе переданных конфигураций.
    Принимает данные в формате List или CSV файл.
    """
    responses = []
    for request in requests:
        model_type = request.config["ml_model_type"]
        model_id = request.config["id"]

        # Инициализация модели
        if model_type == "logistic":
            model = LogisticRegression()
        else:
            responses.append(FitResponse(
                message=f"Model '{model_id}' unsupported type: {model_type}"))
            continue

        try:
            # Проверка, были ли переданы данные как CSV файл
            if file:
                df = pd.read_csv(file.file)
                X = df.iloc[:, :-1].values.tolist()  # Признаки
                y = df.iloc[:, -1].values.tolist()   # Целевая переменная
            else:
                # Если данных в файле нет, используем те, что пришли в теле запроса
                X = request.X
                y = request.y

            # Обучение модели
            model.fit(X, y)
            save_model(model_id, model)

            # Обновление истории
            history = load_model_history()
            history.append({"id": model_id, "type": model_type})
            save_model_history(history)

            responses.append(FitResponse(message=f"Model '{model_id}' trained and saved"))
        except Exception as e:
            responses.append(FitResponse(
                message=f"Failed to train model '{model_id}': {str(e)}"))

    return responses


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