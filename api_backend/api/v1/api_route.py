from fastapi import APIRouter, HTTPException
from api_backend.services.model_history import load_model_history, save_model_history
from sklearn.linear_model import LinearRegression, LogisticRegression
import asyncio
import pickle
import os
from api_backend.serializers.serializers import *

models = {}
router = APIRouter()

@router.post("/fit", response_model=List[FitResponse])
async def fit(requests: List[FitRequest]):
    """
    Обучение и сохранение модели на основе переданных конфигураций.
    Сохраняет модель в хранилище (файловую систему), добавляет запись о модели в историю.
    После обучения модель становится автоматически активной.
    """
    responses = []
    for request in requests:
        model_type = request.config["ml_model_type"]
        model_id = request.config["id"]

        if model_type == "linear":
            model = LinearRegression()
        elif model_type == "logistic":
            model = LogisticRegression()
        else:
            responses.append(FitResponse(
                message=f"Model '{model_id}' unsupported type: {model_type}"))
            continue

        try:
            await asyncio.sleep(60)  # Заглушка для требования к клиентской части дз

            model.fit(request.X, request.y)
            models[model_id] = model

            with open(f"{model_id}.pkl", "wb") as f:
                pickle.dump(model, f)

            history = load_model_history()
            history.append({"id": model_id, "type": model_type})
            save_model_history(history)

            responses.append(FitResponse(message=f"Model '{model_id}' trained and saved"))
        except Exception as e:
            responses.append(FitResponse(
                message=f"Failed to train model '{model_id}': {str(e)}"))

    return responses


@router.post("/load", response_model=List[LoadResponse])
async def load(request: LoadRequest):
    """
    Загрузка модели в пространство инференса (память) из хранилища (файловой системы).
    """
    model_id = request.id
    try:
        with open(f"{model_id}.pkl", "rb") as f:
            models[model_id] = pickle.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    return [LoadResponse(message=f"Model '{model_id}' loaded")]


@router.get("/get_status", response_model=ModelStatusResponse)
async def get_status():
    """
    Получение списка активных моделей в пространстве инференса.
    """
    history = load_model_history()
    active_models = [model_id for model_id in models.keys() if model_id in [entry["id"] for entry in history]]

    return ModelStatusResponse(status="Model Status Ready", models=active_models)


@router.post("/unload", response_model=List[UnloadResponse])
async def unload_model(request: UnloadRequest):
    """
    Удаление модели из пространства инференса (памяти).
    """
    model_id = request.id

    if model_id not in models:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    del models[model_id]

    return [UnloadResponse(message=f"Model '{model_id}' unloaded")]


@router.post("/predict", response_model=List[PredictionResponse])
async def predict(requests: List[PredictRequest]):
    """
    Создание предсказаний для данных с выбором активных моделей на платформе инференса.
    """
    responses = []
    for request in requests:
        model_id = request.id
        if model_id not in models:
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

        model = models[model_id]
        await asyncio.sleep(5)  # Заглушка для демонстрации асинхронности
        predictions = model.predict(request.X).tolist()
        responses.append(PredictionResponse(predictions=predictions))

    return responses


@router.get("/list_models", response_model=List[ModelListResponse])
async def list_models():
    """
    Возвращает список всех моделей (активных и заархивированных).
    """
    model_type_map = {
        LinearRegression: "linear",
        LogisticRegression: "logistic"
    }

    models_in_memory = [
        ModelMetadata(id=model_id, type=model_type_map.get(type(model), "unknown"))
        for model_id, model in models.items()
    ]

    history = load_model_history()
    archive_models = []
    for entry in history:
        model_id = entry["id"]
        model_type = entry["type"]
        if model_id not in models and os.path.exists(f"{model_id}.pkl"):
            archive_models.append(ModelMetadata(id=model_id, type=model_type))

    all_models = models_in_memory + archive_models
    return [ModelListResponse(models=all_models)]



@router.delete("/remove/{model_id}", response_model=List[RemoveResponse])
async def remove(model_id: str):
    """
    Удаление модели по ее идентификатору из памяти, файловой системы и истории обученных моделей.
    """
    if model_id in models:
        del models[model_id]

    history = load_model_history()
    history = [entry for entry in history if entry["id"] != model_id]
    save_model_history(history)

    file_path = f"{model_id}.pkl"
    if os.path.exists(file_path):
        os.remove(file_path)
        return [RemoveResponse(message=f"Model '{model_id}' removed.")]
    else:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found in storage.")


@router.delete("/remove_all", response_model=List[RemoveResponse])
async def remove_all():
    """
    Удаление всех моделей из памяти, файловой системы и истории обученных моделей.
    """
    removed_models = []
    active_models = list(models.keys())
    processed_models = set()

    for model_id in active_models:
        if model_id in models:
            del models[model_id]
        file_path = f"{model_id}.pkl"
        if os.path.exists(file_path):
            os.remove(file_path)
        removed_models.append(RemoveResponse(message=f"Model '{model_id}' removed"))
        processed_models.add(model_id)

    model_files = [file for file in os.listdir() if file.endswith(".pkl")]
    for file in model_files:
        model_id = file[:-4]
        if model_id not in processed_models:
            os.remove(file)
            removed_models.append(RemoveResponse(message=f"Model '{model_id}' removed"))
            processed_models.add(model_id)

    save_model_history([])

    if not removed_models:
        raise HTTPException(status_code=404, detail="No models found to remove")

    return removed_models