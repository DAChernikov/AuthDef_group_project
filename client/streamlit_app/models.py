import streamlit as st

import pandas as pd
import requests
import logging

from constants import API_URL

logger = logging.getLogger(__name__)


def models():
    """
    Отображает список доступных моделей и позволяет выполнять операции с ними.
    """

    st.header("Список моделей")

    try:
        response = requests.get(f"{API_URL}/api/v1/models/list_models")
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error(e)
        st.error("Ошибка при получении данных")
        return

    models_data = []

    response_json = response.json()
    if not response_json[0]['models']:
        st.warning("Нет доступных моделей")
        return

    for entry in response_json:
        for model in entry["models"]:
            model_info = {
                "Имя модели": model["id"],
            }

            if not model['metrics']:
                model_info['Вид'] = 'Готовая'
            else:
                model_info["Тип"] = model["type"]
                model_info["Accuracy"] = model["metrics"]["accuracy"]
                model_info["Precision"] = model["metrics"]["precision"]
                model_info["Recall"] = model["metrics"]["recall"]
                model_info["F1 Score"] = model["metrics"]["f1"]
                model_info['Вид'] = 'Обученная'

            models_data.append(model_info)

    df = pd.DataFrame(models_data)
    st.dataframe(df)

    st.header("Операции с моделями")

    selected_id = st.selectbox("Выберите модель", df['Имя модели'])

    protected_models = ["base_model"]

    if selected_id in protected_models:
        st.warning(f"Модель '{selected_id}' не может быть удалена.")
    else:
        if st.button(f"Удалить {selected_id}"):
            try:
                response = requests.delete(f"{API_URL}/api/v1/models/remove/{selected_id}")
                response.raise_for_status()
                st.success(f"Модель {selected_id} успешно удалена!")
            except requests.exceptions.RequestException as e:
                st.error(f"Ошибка при удалении модели: {str(e)}")
