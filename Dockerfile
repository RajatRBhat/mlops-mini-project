FROM python:3.12-slim

WORKDIR /app

COPY fastapi_api/ /app/

COPY models/vectorizer.pkl /app/models/vectorizer.pkl

COPY models/model.pkl /app/models/model.pkl

RUN pip install -r requirements.txt

RUN python -m nltk.downloader stopwords wordnet

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "60"]