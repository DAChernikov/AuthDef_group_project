from pydantic import BaseModel
from typing import List, Dict, Union

"""Характеристика pydantic моделей, используемых в работе сервиса"""

class ModelMetadata(BaseModel):
    """Метаданные модели, включая идентификатор и тип."""
    id: str
    type: str

class ModelConfig(BaseModel):
    """Конфигурация модели с гиперпараметрами."""
    id: str
    ml_model_type: str
    hyperparameters: Dict[str, Union[str, int, bool, float]]

class ModelStatusResponse(BaseModel):
    """Ответ сервиса с текущим статусом и списком активных моделей."""
    status: str
    models: List[str]

class ModelListResponse(BaseModel):
    """Список всех моделей, доступных в системе."""
    models: List[ModelMetadata]

class RemoveResponse(BaseModel):
    """Ответ о статусе удаления модели."""
    message: str

class FitRequest(BaseModel):
    """Запрос на обучение модели с данными и конфигурацией."""
    X: list
    y: list
    config: dict

class FitResponse(BaseModel):
    """Ответ о статусе обучения модели."""
    message: str

class LoadRequest(BaseModel):
    """Запрос на загрузку модели по идентификатору."""
    id: str

class LoadResponse(BaseModel):
    """Ответ о статусе загрузки модели."""
    message: str

class UnloadRequest(BaseModel):
    """Запрос на выгрузку модели из памяти."""
    id: str

class UnloadResponse(BaseModel):
    """Ответ о статусе выгрузки модели."""
    message: str

class PredictRequest(BaseModel):
    """Запрос на предсказание, включающий идентификатор модели и входные данные."""
    id: str
    X: List[List[float]]

class PredictionResponse(BaseModel):
    """Ответ с предсказаниями модели."""
    predictions: List[float]