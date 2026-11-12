import streamlit as st
import requests
from PIL import Image

st.set_page_config(
    page_title="Shapes Project",
    page_icon="",
    layout="centered"
)

API_URL = "http://127.0.0.1:8000"

st.title("Shapes Recognition")
st.write("Загрузите изображение геометрической фигуры для распознавания.")

try:
    response = requests.get(f"{API_URL}/classes", timeout=5)

    if response.status_code == 200:
        classes = response.json()["classes"]
        st.info(f"Модель распознаёт: {', '.join(classes)}")

except requests.exceptions.RequestException:
    st.warning("Backend не запущен.")

uploaded_file = st.file_uploader(
    "Выберите изображение",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Загруженное изображение",
            use_container_width=True
        )

        if st.button("Распознать фигуру", use_container_width=True):
            with st.spinner("Модель анализирует изображение..."):
                response = requests.post(
                    f"{API_URL}/predict",
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type
                        )
                    },
                    timeout=30
                )

            if response.status_code == 200:
                result = response.json()

                st.success("Распознавание завершено!")
                st.subheader(f"Фигура: {result['prediction']}")
                st.metric(
                    "Уверенность модели",
                    f"{result['confidence']}%"
                )
                st.progress(result["confidence"] / 100)

            else:
                st.error(response.json().get("detail", "Ошибка распознавания"))

    except requests.exceptions.ConnectionError:
        st.error("Backend не запущен!")

    except requests.exceptions.Timeout:
        st.error("Время ожидания ответа истекло.")

    except Exception:
        st.error("Не удалось обработать изображение.")