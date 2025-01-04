import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error , r2_score

# Load the dataset
df = pd.read_csv('CarsDataset.csv', encoding='cp1252')
st.title('CS CAPSTONE PROJECT')

st.subheader('Data Information:')

# Boxplot for grouped car prices
fig, ax = plt.subplots()
ax.boxplot(df['Car Prices$'], vert=False, patch_artist=True,
           boxprops=dict(facecolor='skyblue', color='black'),
           medianprops=dict(color='red'))
ax.set_title('Distribution of Car Prices')
ax.set_xlabel('Price (in $)')
st.pyplot(fig)

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
fig, ax = plt.subplots()
ax.scatter(df['HorsePower'], df['Car Prices$'], alpha=0.5)
ax.set_title('Price vs. Horsepower')
ax.set_xlabel('Horsepower')
ax.set_ylabel('Price (in $)')
price_ticks = np.arange(0, df['Car Prices$'].max(), 25000)
horsepower_ticks = np.arange(0, df['HorsePower'].max() + 50, 150)
ax.set_xticks(horsepower_ticks)
ax.set_yticks(price_ticks)
st.pyplot(fig)

#The model uses a log transformation to better handle price variability and improve prediction accuracy.
#All displayed prices have been converted back to the original scale for ease of interpretation.
X = df[['HorsePower', 'Torque(Nm)']]
df['Log Price'] = np.log(df['Car Prices$'])
y = df['Log Price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)
# Predict on test data
y_pred = model.predict(X_test)
# Calculate RMSE
rmse = root_mean_squared_error(y_test, y_pred)
# Calculate R² score
r2 = r2_score(y_test, y_pred)
# Display model accuracy metrics
st.subheader("Model Accuracy")
st.write(f"**Root Mean Squared Error (RMSE):** {rmse:,.2f}")
st.write(f"**R² Score:** {r2:.2f}")
st.write("""
- **RMSE** gives an idea of how much the predicted prices deviate from actual prices on average.
- **R² Score** indicates how much variance in the price is explained by the model, with a value closer to 1 being ideal.
""")

st.subheader('Tool for Predicting Car Prices')
st.write('Please enter the HorsePower and Torque values of the car you would like to predict the price for and then press the predict price button. (Range: 50-1300 for HorsePower, 50-3500 for Torque)')
# Input fields for prediction
horsepower = st.number_input('HorsePower', min_value=50, max_value=1300, step=10)
torque = st.number_input('Torque(Nm)', min_value=50, max_value=3500, step=10)

# Predict log-transformed price
if st.button('Predict Price'):
    user_input = np.array([[horsepower, torque]])
    log_price_pred = model.predict(user_input)[0]
    price_pred = np.exp(log_price_pred)
    lower_bound = price_pred * 0.75
    upper_bound = price_pred * 1.25
    st.write(f"Predicted Price: ${price_pred:,.2f}")
    st.write(f"Price Ranges: ${lower_bound:,.2f} - ${upper_bound:,.2f}")