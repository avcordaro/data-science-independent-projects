import re
import pickle
from flask import Flask, request

model = None
cv = None
sp = None
app = Flask(__name__)


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
def get_prediction():
    data = request.data.decode('UTF-8')
    processed_text = preprocess(data)
    pred = model.predict(processed_text)[0]
    return "Negative" if pred == 0 else "Positive"


if __name__ == '__main__':
    load_objects() 
    app.run(host='127.0.0.1', port=80)