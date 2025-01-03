import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('CarsDataset.csv', encoding='cp1252')

st.title('CS CAPSTONE PROJECT')

st.subheader('Data Information:')
st.write(df.describe())

# Pie chart of car manufacturers
manufacturer_counts = df['Company Names'].value_counts()
threshold = 0.02 * manufacturer_counts.sum()
manufacturer_counts['Other'] = manufacturer_counts[manufacturer_counts < threshold].sum()
manufacturer_counts = manufacturer_counts[manufacturer_counts >= threshold]
fig, ax = plt.subplots()
ax.pie(manufacturer_counts, labels=manufacturer_counts.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
ax.set_title('Car Distribution by Manufacturer')
st.pyplot(fig)

# Scatter plot for Price vs Horsepower
price_threshold = 1000000
filtered_df = df[df['Car Prices$'] <= price_threshold]
fig, ax = plt.subplots()
ax.scatter(filtered_df['HorsePower'], filtered_df['Car Prices$'], alpha=0.5)
ax.set_title('Price vs. Horsepower')
ax.set_xlabel('Horsepower')
ax.set_ylabel('Price (in $)')
price_ticks = np.arange(0, filtered_df['Car Prices$'].max(), 50000)
horsepower_ticks = np.arange(0, filtered_df['HorsePower'].max() + 50, 150)
ax.set_xticks(horsepower_ticks)
ax.set_yticks(price_ticks)
st.pyplot(fig)
