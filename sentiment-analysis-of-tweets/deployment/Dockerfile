FROM python:3.6-slim
COPY ./main.py /deploy/
COPY ./requirements.txt /deploy/
COPY ./log_regr_model.pkl /deploy/
COPY ./count_vectorizer.pkl /deploy/
COPY ./select_percentile.pkl /deploy/
WORKDIR /deploy/
RUN pip install -r requirements.txt
EXPOSE 80
CMD ["python", "main.py"]