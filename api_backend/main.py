import os
import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict
from api.v1.api_route import router as model_router
from services import model_history

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(BASE_DIR, "models_storage")
HISTORY_FILE = os.path.join(MODEL_DIR, "model_history.json")
os.makedirs(MODEL_DIR, exist_ok=True)

# Инициализация путей в модулях, где они (пути) требуются
model_history.init_paths(BASE_DIR)

# Инициализация предзагруженной модели baseline из models_storage в регистре моделей
history = model_history.load_model_history()

# Подготовка метрик для корректности
accuracy = 0.90
precision = 0.90
recall = 0.91
f1 = 0.91
conf_matrix = np.array([
    [16, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0],
    [0, 44,  0,  1,  0,  0,  1,  0,  0,  0,  1,  1,  1],
    [0,  0, 36,  1,  0,  0,  1,  0,  0,  0,  1,  1,  0],
    [1,  0,  1, 35,  0,  0,  3,  0,  1,  0,  0,  0,  3],
    [0,  0,  0,  0, 11,  0,  0,  0,  0,  0,  0,  0,  1],
    [0,  1,  0,  0,  0, 54,  0,  0,  1,  0,  0,  0,  0],
    [0,  0,  1,  0,  1,  0, 63,  0,  0,  0,  0,  0,  1],
    [0,  0,  0,  1,  0,  0,  0,  8,  0,  0,  0,  0,  0],
    [0,  0,  0,  1,  0,  0,  0,  0, 50,  0,  0,  1,  1],
    [1,  0,  0,  0,  1,  0,  0,  0,  0, 31,  0,  1,  0],
    [1,  0,  0,  0,  0,  0,  0,  0,  1,  0, 34,  1,  0],
    [0,  0,  1,  1,  1,  0,  1,  0,  1,  1,  0, 59,  0],
    [0,  2,  0,  1,  0,  0,  1,  0,  0,  0,  0,  1, 21]
])
# Преобразуем confusion matrix в список для JSON
conf_matrix_list = conf_matrix.tolist()

# Добавление информации о главной модели сервиса в регистр
existing_model = next((item for item in history if item['id'] == 'base_model'), None)
if not existing_model:
    history.append({
        "id": 'base_model',
        "type": 'XGBoost',
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": conf_matrix_list
        }
    })
    model_history.save_model_history(history)

# Запуск приложения
app = FastAPI(
    title="api_backend",
    docs_url="/api/openapi",
    openapi_url="/api/openapi.json",
)


class StatusResponse(BaseModel):
    status: str

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"status": "App healthy"}]}
    )


@app.get("/")
async def root():
    """
    Возвращает статус о готовности приложения.
    """
    return [{"status": "App ready"}]


app.include_router(model_router, prefix="/api/v1/models")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
