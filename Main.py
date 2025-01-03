import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('CarsDataset.csv', encoding='cp1252')

st.title('CS CAPSTONE PROJECT')

st.subheader('Data Information:')
st.write(df.describe())