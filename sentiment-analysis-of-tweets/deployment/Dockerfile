FROM python:3.8-slim

COPY ./main.py /deploy/
COPY ./requirements.txt /deploy/
COPY ./log_regr_model.pkl /deploy/
COPY ./count_vectorizer.pkl /deploy/
COPY ./select_percentile.pkl /deploy/

WORKDIR /deploy/

RUN pip install -r requirements.txt

CMD ["python", "main.py"]