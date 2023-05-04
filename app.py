# importing required libraries

from extract_feature import FeatureExtraction
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn import metrics
import warnings
import pickle
warnings.filterwarnings('ignore')

file = open(r"C:\Users\vedant raikar\Desktop\phishing website detection\model\trained_model.pkl", "rb")
gbc = pickle.load(file)
file.close()


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    pred = None 
    if request.method == "POST":

        url = request.form["url"]
        obj = FeatureExtraction(url)
        x = np.array(obj.getFeaturesList()).reshape(1, 30)

        y_pred = gbc.predict(x)[0]
        # 1 is safe
        # -1 is unsafe
        pred = y_pred
  
    return render_template('index.html', pred=pred)


if __name__ == "__main__":
    app.run(debug=True)
