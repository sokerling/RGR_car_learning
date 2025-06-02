import streamlit as st
import pandas as pd


st.title("Описание датасета")

st.markdown("""
**Предметная область:** Недвижимость в Мумбаи, Индия  
**Цель:** Предсказание цены квартиры на основе характеристик недвижимости  
""")

df = pd.read_csv("data/mumbai_houses_task_final.csv")
st.write("Фрагмент данных:")
st.dataframe(df.head())

st.markdown("### Описание признаков:")
st.markdown("""
- `price`: Цена недвижимости (целевая переменная)  
- `area`: Площадь квартиры (в квадратных футах)  
- `latitude`, `longitude`: Географические координаты  
- `Bedrooms`, `Bathrooms`, `Balcony`: Кол-во комнат  
- `Status`: Статус объекта  
- `neworold`: Новое или вторичное жилье  
- `parking`, `Furnished_status`, `Lift`: Бинарные признаки комфорта  
- `type_of_building`: Тип здания  
""")