import pickle
from flask import Flask, render_template, request, send_from_directory, redirect, url_for
from model import run_model
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

@app.route("/")
def main_page():
    return render_template("form.html")

@app.route("/predict", methods=["POST"])
def predict():
    ticker = request.form['ticker']
    prediction, date = run_model(str(ticker))
    image_file = f'{ticker}_stock_price.jpeg'
    return render_template("display_image.html", image_file = image_file, result=f"Share price for {ticker} on {date} is {prediction}")


if __name__ == '__main__':
    app.run(debug=True)
