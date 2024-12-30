import streamlit as st

import requests

from constants import API_URL


def inference():
    st.header('Определение авторства текста')

    try:
        response = requests.get(f"{API_URL}/api/v1/models/list_models")
        response.raise_for_status()
        models = response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Ошибка при получении моделей: {e}")
        models = []

    if models[0]['models']:
        model_ids = [model['id'] for model in models[0]['models']]
        selected_model_id = st.selectbox("Выберите модель", model_ids)
    else:
        st.warning("Не удалось загрузить модели.")
        selected_model_id = None

    if selected_model_id:
        user_input = st.text_area("Введите текст:")

        if st.button("Отправить"):
            if user_input:
                payload = [{  # Формируем объект, а не список
                    "id": selected_model_id,
                    "X": [user_input],  # Оборачиваем текст в список, как ожидает API
                }]

                try:
                    response = requests.post(f"{API_URL}/api/v1/models/predict_request", json=payload)
                    if response.status_code == 200:
                        st.success("Результат:")
                        st.json(response.json())
                    else:
                        st.error(f"Ошибка: {response.status_code}\n{response.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Ошибка при отправке запроса: {e}")
            else:
                st.warning("Пожалуйста, введите текст.")
