import streamlit as st

import pandas as pd
import requests

from constants import API_URL


def models():
    st.header("Загрузка модели")

    uploaded_file = st.file_uploader("Загрузить готовую модель", type=['pkl'])
    uploaded_file_w2v = st.file_uploader("Загрузить w2v модель", type=['pkl'])

    if uploaded_file is not None and uploaded_file_w2v is not None:
        file_content = uploaded_file.read()

        files = {
            'file': (uploaded_file.name, file_content, uploaded_file.type)
        }

        file_content_w2v = uploaded_file_w2v.read()

        files_w2v = {
            'file': (uploaded_file_w2v.name, file_content_w2v, uploaded_file_w2v.type)
        }

        try:
            response = requests.post(f"{API_URL}/api/v1/models/save_model", files=files)
            response.raise_for_status()
            st.success(f"Модель {uploaded_file.name} успешно загружена!")
        except requests.exceptions.RequestException as e:
            st.error(f"Ошибка при сохранении модели: {str(e)}")

        try:
            response = requests.post(f"{API_URL}/api/v1/models/save_model", files=files_w2v)
            response.raise_for_status()
            st.success(f"Модель {uploaded_file_w2v.name} успешно загружена!")
        except requests.exceptions.RequestException as e:
            st.error(f"Ошибка при сохранении модели: {str(e)}")

    st.header("Список моделей")

    try:
        response = requests.get(f"{API_URL}/api/v1/models/list_models")
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        st.error(f"Ошибка при получении данных: {str(e)}")
        return

    models_data = []

    if not response.json()[0]['models']:
        st.warning("Нет доступных моделей")
        return

    for entry in response.json():
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

    if st.button(f"Удалить {selected_id}"):
        try:
            response = requests.delete(f"{API_URL}/api/v1/models/remove/{selected_id}")
            response.raise_for_status()
            st.success(f"Модель {selected_id} успешно удалена!")
        except requests.exceptions.RequestException as e:
            st.error(f"Ошибка при удалении модели: {str(e)}")
