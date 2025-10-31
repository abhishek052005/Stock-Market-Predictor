import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load your trained model
model = load_model(r"C:\Users\Abhi tiwari\Documents\project\Stock_prediction\Stock_Prediction Model.keras")

# Streamlit header
st.header('📈 Stock Market Predictor ')

# Stock input
stock = st.text_input('Enter Stock Symbol', 'GOOG')
start = '2014-01-01'
end   = '2022-12-22'

# Download stock data
data = yf.download(stock, start, end)

# Show raw data
st.subheader('Stock Data (Closing Price)')
st.write(data.tail())

# Prepare dataframe with only 'Close'
df = data[['Close']].copy()

# Split train/test
train_size = int(len(df) * 0.8)
train_data = df[:train_size].values
test_data  = df[train_size:].values

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df.values)

# --- Create sequences ---
def create_sequences(dataset, time_step=60):
    x, y = [], []
    for i in range(time_step, len(dataset)):
        x.append(dataset[i-time_step:i, 0])
        y.append(dataset[i, 0])
    return np.array(x), np.array(y)

time_step = 60
x_test, y_test = create_sequences(scaled_data[train_size - time_step:], time_step)

# Reshape for LSTM
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

# Predict with model
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)  # back to original scale
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot predictions vs original
st.subheader("Predicted vs Original Stock Price")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(y_test_rescaled, label="Original Price", color='blue')
ax.plot(predictions, label="Predicted Price", color='red')
ax.set_xlabel("Time")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

# Future prediction (next day)
last_60_days = scaled_data[-time_step:]
last_60_days = last_60_days.reshape((1, time_step, 1))
next_pred = model.predict(last_60_days)
next_pred = scaler.inverse_transform(next_pred)

st.subheader("Next Day Prediction")
st.write(f"Predicted Closing Price: **{next_pred[0][0]:.2f}**")




