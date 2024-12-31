import json
import os
import pickle
from fastapi import HTTPException

BASE_DIR = None
MODEL_DIR = None
HISTORY_FILE = None


def init_paths(base_dir):
    """
    Инициализирует пути для модели и истории.
    """

    global BASE_DIR, MODEL_DIR, HISTORY_FILE
    BASE_DIR = base_dir
    MODEL_DIR = os.path.join(BASE_DIR, "models_storage")
    HISTORY_FILE = os.path.join(MODEL_DIR, "model_history.json")
    os.makedirs(MODEL_DIR, exist_ok=True)


def load_model_history():
    """
    Загружает историю моделей из файла-регистратора.
    """

    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []


def save_model_history(history):
    """
    Сохраняет историю моделей в файл-регистратор моделей.
    """

    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)


def save_model(model_id, model):
    """
    Сохраняет модель в файловой системе.
    """

    model_path = os.path.join(MODEL_DIR, f"{model_id}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)


def load_model(model_id):
    """
    Загружает модель из файловой системы.
    """

    model_path = os.path.join(MODEL_DIR, f"{model_id}.pkl")
    print('Loading model from: ', model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model '{model_id}' not found")
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print(f"Model loaded: {type(model)}")
        return model
    except Exception as e:
        print(f"Error loading model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}") from e


def delete_model(model_id):
    """
    Удаляет модель из файловой системы.
    """

    model_path = os.path.join(MODEL_DIR, f"{model_id}.pkl")
    if os.path.exists(model_path):
        os.remove(model_path)
