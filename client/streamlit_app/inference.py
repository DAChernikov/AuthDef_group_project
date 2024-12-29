import streamlit as st

import requests
import logging

from constants import API_URL

logger = logging.getLogger(__name__)


def inference():
    st.header('Определение авторства теста')

    try:
        response = requests.get(f"{API_URL}/api/v1/models/list_models")
        response.raise_for_status()
        models = response.json()
    except requests.exceptions.RequestException as e:
        logger.error(e)
        st.error("Ошибка при получении данных")
        return

    if models[0]['models']:
        model_ids = [model['id'] for model in models[0]['models']]
        selected_model_id = st.selectbox("Выберите модель", model_ids)
    else:
        logger.warning("Cannot load the model")
        st.warning("Не удалось загрузить модели.")
        selected_model_id = None

    if selected_model_id:
        user_input = st.text_area("Введите текст:")

        if st.button("Отправить"):
            if user_input:
                payload = [
                    {
                        'text': user_input,
                        'model_id': selected_model_id,
                    }
                ]

                try:
                    response = requests.post(f"{API_URL}/api/v1/models/predict_request", json=payload)
                    if response.status_code == 200:
                        st.success("Результат:")
                        st.json(response.json())
                    else:
                        st.error(f"Ошибка: {response.status_code}")
                except requests.exceptions.RequestException as e:
                    logger.error(e)
                    st.error("Ошибка при получении данных")
            else:
                st.warning("Пожалуйста, введите текст.")
