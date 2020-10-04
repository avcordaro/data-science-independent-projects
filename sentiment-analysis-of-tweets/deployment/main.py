import re
import os
import pickle
from flask import Flask, request
from flask_cors import CORS, cross_origin

model = None
cv = None
sp = None
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

def load_objects():
    global model, cv, sp
    model = pickle.load(open('log_regr_model.pkl', 'rb'))
    cv = pickle.load(open('count_vectorizer.pkl', 'rb'))
    sp = pickle.load(open('select_percentile.pkl', 'rb'))

def preprocess(text):
    cleaned_text = re.sub("@\S+|#\S+|\d+|https?:\S+|http?:\S+", '', str(text))
    text_cv = cv.transform([cleaned_text])
    text_cv_small = sp.transform(text_cv)
    return text_cv_small


@app.route('/', methods=['POST'])
@cross_origin()
def get_prediction():
    data = request.data.decode('UTF-8')
    processed_text = preprocess(data)
    pred = model.predict(processed_text)[0]
    return "Negative" if pred == 0 else "Positive"


if __name__ == '__main__':
    load_objects() 
    app.run(host='127.0.0.1', port=5000)