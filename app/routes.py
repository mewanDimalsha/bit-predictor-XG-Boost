from flask import jsonify
from app import app
import pandas as pd
import yfinance as yf
import pickle
import numpy as np


def load_model():
    with open("model/model.pkl", "rb") as file:
        model, scaler = pickle.load(file)
    return model, scaler


def predict_next_day(model, data, scaler, time_step=15):
    data = scaler.transform(np.array(data).reshape(-1, 1))
    X_input = data[-time_step:].reshape(1, -1)
    X_input = X_input.reshape((1, time_step))
    y_pred = model.predict(X_input)
    return scaler.inverse_transform(y_pred.reshape(-1, 1))


@app.route("/predict", methods=["GET"])
def predict():
    # Get the latest data
    bitcoindf = yf.Ticker("BTC-USD").history(period="max")
    bitcoindf.index = pd.to_datetime(bitcoindf.index)
    bitcoindf = bitcoindf[["Close"]]

    # Load the model and scaler
    model, scaler = load_model()

    # Predict the next day close price
    next_day_price = predict_next_day(model, bitcoindf["Close"], scaler)
    return jsonify({"next_day_close_price": float(next_day_price[0][0])})
