from flask import Flask, render_template, jsonify
import pandas as pd
import yfinance as yf
import pickle
import numpy as np
from apscheduler.schedulers.background import BackgroundScheduler
import joblib
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error

app = Flask(__name__)


def load_model():
    with open("model/model.pkl", "rb") as file:
        model, scaler = joblib.load(file)
    return model, scaler


def predict_next_day(model, data, scaler, time_step=15):
    data = scaler.transform(np.array(data).reshape(-1, 1))
    X_input = data[-time_step:].reshape(1, -1)
    X_input = X_input.reshape((1, time_step))
    y_pred = model.predict(X_input)
    return scaler.inverse_transform(y_pred.reshape(-1, 1))


@app.route("/predict", methods=["GET"])
def predict():
    bitcoindf = yf.Ticker("BTC-USD").history(period="max")
    bitcoindf.index = pd.to_datetime(bitcoindf.index)
    bitcoindf = bitcoindf[["Close"]]
    model, scaler = load_model()
    next_day_price = predict_next_day(model, bitcoindf["Close"], scaler)
    return render_template("result.html", next_day_close_price=next_day_price[0][0])


@app.route("/")
def index():
    return "Welcome to the Bitcoin Price Prediction API. Use the /predict endpoint to get predictions."


def load_data():
    ticker = "BTC-USD"
    bitcoindf = yf.Ticker(ticker).history(period="max")
    bitcoindf.index = pd.to_datetime(bitcoindf.index)
    bitcoindf.drop(columns=["Dividends", "Stock Splits"], inplace=True)
    closedf = bitcoindf[["Close"]]
    closedf = closedf[closedf.index > "2021-04-13"]
    return closedf


def preprocess_data(closedf):
    scaler = MinMaxScaler(feature_range=(0, 1))
    closedf["Close"] = scaler.fit_transform(np.array(closedf["Close"]).reshape(-1, 1))
    return closedf, scaler


def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i : (i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)


def train_and_save_model():
    closedf = load_data()
    closedf, scaler = preprocess_data(closedf)
    training_size = int(len(closedf) * 0.70)
    test_size = len(closedf) - training_size
    train_data, test_data = closedf[:training_size], closedf[training_size:]
    time_step = 15
    X_train, y_train = create_dataset(train_data.values, time_step)
    X_test, y_test = create_dataset(test_data.values, time_step)
    my_model = XGBRegressor(n_estimators=1000)
    my_model.fit(X_train, y_train, verbose=False)
    predictions = my_model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = math.sqrt(mean_squared_error(y_test, predictions))
    print("Mean Absolute Error - MAE : " + str(mae))
    print("Root Mean Squared Error - RMSE : " + str(rmse))
    joblib.dump((my_model, scaler), "model/model.pkl")


scheduler = BackgroundScheduler()
scheduler.add_job(train_and_save_model, "cron", hour=0, minute=0)
scheduler.start()

if __name__ == "__main__":
    train_and_save_model()
    app.run(debug=True)
