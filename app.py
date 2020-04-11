import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
import joblib
vectorizer_keyword=joblib.load('artifacts/vectorizer_keyword.pkl')
vectorizer_location=joblib.load('artifacts/vectorizer_location.pkl')
tfidf_vectorizer=joblib.load('artifacts/tfidf_vectorizer.pkl')
bst=joblib.load('artifacts/model.pkl')

from flask import Flask,render_template,request,url_for
app = Flask(__name__)


#
@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/index')
def form_input():
    return render_template('index.html')

@app.route('/predict',methods=['post'])
def predict():
    location=request.form['location']
    keyword=request.form['keyword']
    message=request.form['message']

    location_transform=vectorizer_location.transform([location])
    keyword_transform=vectorizer_keyword.transform([keyword])
    message_transform=tfidf_vectorizer.transform([message])

    X_query=hstack([location_transform,keyword_transform,message_transform])
    X_query = xgb.DMatrix(X_query)
    pred=bst.predict(X_query)
    if(pred>0.5):
        return 'It is a Disaster tweet'
    else:
        return 'The tweet is not related to diaster category'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
