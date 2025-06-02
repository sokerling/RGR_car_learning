import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка датасета
@st.cache_data
def load_data():
    return pd.read_csv('data/mumbai_houses_task_final.csv')

data = load_data()

st.title("Визуализация зависимостей в наборе данных")

# 1. Гистограмма распределения цены
st.subheader("Распределение цены домов")
fig1, ax1 = plt.subplots()
sns.histplot(data['price'], bins=30, kde=True, ax=ax1)
ax1.set_xlabel("Цена (рупий)")
ax1.set_ylabel("Количество")
st.pyplot(fig1)

# 2. Scatter plot: площадь vs цена
st.subheader("Площадь vs Цена")
fig2, ax2 = plt.subplots()
sns.scatterplot(x='area', y='price', data=data, ax=ax2)
ax2.set_xlabel("Площадь (кв. футов)")
ax2.set_ylabel("Цена (рупий)")
st.pyplot(fig2)

# 3. Boxplot цены по количеству комнат
st.subheader("Цена в зависимости от количества комнат")
fig3, ax3 = plt.subplots()
sns.boxplot(x='Bedrooms', y='price', data=data, ax=ax3)
ax3.set_xlabel("Количество комнат")
ax3.set_ylabel("Цена (рупий)")
st.pyplot(fig3)

# 4. Тепловая карта корреляций для выбранных количественных признаков
st.subheader("Корреляционная матрица признаков: price, area, latitude, longitude")

cols = ['price', 'area', 'latitude', 'longitude']
data_subset = data[cols]

fig4, ax4 = plt.subplots(figsize=(6, 5))
sns.heatmap(data_subset.corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax4)
st.pyplot(fig4)