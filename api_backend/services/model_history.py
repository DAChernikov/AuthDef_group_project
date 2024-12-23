import json
import os
import pickle

HISTORY_FILE = "model_history.json"  # Имя и путь файла-регистратора моделей
MODEL_DIR = "models_storage"
os.makedirs(MODEL_DIR, exist_ok=True)

def load_model_history():
    """Загружает историю моделей из файла-регистратора."""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_model_history(history):
    """Сохраняет историю моделей в файл-регистратор моделей."""
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

def save_model(model_id, model):
    """Сохраняет модель в файловой системе."""
    model_path = os.path.join(MODEL_DIR, f"{model_id}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

def load_model(model_id):
    """Загружает модель из файловой системы."""
    model_path = os.path.join(MODEL_DIR, f"{model_id}.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model '{model_id}' not found")
    with open(model_path, "rb") as f:
        return pickle.load(f)

def delete_model(model_id):
    """Удаляет модель из файловой системы."""
    model_path = os.path.join(MODEL_DIR, f"{model_id}.pkl")
    if os.path.exists(model_path):
        os.remove(model_path)