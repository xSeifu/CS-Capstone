import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('CarsDataset.csv')

st.title('CS CAPSTONE PROJECT')

st.subheader('Data Information:')
st.write(df.describe())