from pydantic import BaseModel
from typing import List, Dict, Union
import pandas as pd

"""Характеристика pydantic моделей, используемых в работе сервиса"""

class DataProcessor:
    @staticmethod
    def parse_data(X: Union[str, list], y: Union[str, list] = None):
        """
        Парсит входные данные (CSV, список) в DataFrame и Series.
        """
        if isinstance(X, str):
            X = pd.read_csv(X)
        else:
            X = pd.DataFrame(X)

        if y:
            y = pd.Series(y)
        return X, y

class ModelMetadata(BaseModel):
    """Метаданные модели, включая идентификатор, тип и метрики."""
    id: str
    type: str
    metrics: Dict[str, Union[float, List[List[int]]]] = {}

class ModelConfig(BaseModel):
    """Конфигурация модели с гиперпараметрами."""
    id: str
    ml_model_type: str
    hyperparameters: Dict[str, Union[str, int, bool, float]] = {}

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
    y: Union[list, None] = None
    config: ModelConfig

class FitResponse(BaseModel):
    """Ответ о статусе обучения модели."""
    message: str

class SaveResponse(BaseModel):
    """Ответ о статусе сохранения модели."""
    message: str

class PredictRequest(BaseModel):
    """Запрос на предсказание, включающий идентификатор модели и входные данные."""
    id: str
    X: List[str]

class PredictionResponse(BaseModel):
    """Ответ с предсказаниями модели."""
    predictions: List[str]
