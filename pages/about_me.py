import streamlit as st
from PIL import Image


st.title("О разработчике")
st.subheader("ФИО: Мельников Максим Кириллович")
st.text("Группа: ФИТ-231")
st.markdown("**Тема РГР:** Разработка Web-приложения (дашборда) для инференса моделей ML и анализа данных")

image = Image.open("images/MaxOn.jpg")
st.image(image, caption='Мельников Максим', use_container_width=True)