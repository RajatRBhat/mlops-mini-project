from fastapi import FastAPI
from pydantic import BaseModel

import pickle
import pandas as pd

import numpy as np
import pandas as pd
import os
import re
import nltk
import mlflow
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = FastAPI()

class UserInput(BaseModel):
    user_query : str

# Executing uvicorn fastapi_api.app:app --reload

def lemmatization(text):
    """Lemmatize the text."""
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

def remove_stop_words(text):
    """Remove stop words from the text."""
    stop_words = set(stopwords.words("english"))
    text = [word for word in str(text).split() if word not in stop_words]
    return " ".join(text)

def removing_numbers(text):
    """Remove numbers from the text."""
    text = ''.join([char for char in text if not char.isdigit()])
    return text

def lower_case(text):
    """Convert text to lower case."""
    text = text.split()
    text = [word.lower() for word in text]
    return " ".join(text)

def removing_punctuations(text):
    """Remove punctuations from the text."""
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "")
    text = re.sub('\s+', ' ', text).strip()
    return text

def removing_urls(text):
    """Remove URLs from the text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df):
    """Remove sentences with less than 3 words."""
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan

def normalize_text(text):
    text = lower_case(text)
    text = remove_stop_words(text)
    text = removing_numbers(text)
    text = removing_punctuations(text)
    text = removing_urls(text)
    text = lemmatization(text)

    return text

def get_latest_model_version(model_name, stage="Staging"):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    client = mlflow.MlflowClient()
    latest_version = client.get_latest_versions(model_name)
    return latest_version[0].version if latest_version else None

def load_model():
    new_model_name = "my_model"
    new_model_version = get_latest_model_version(new_model_name)
    new_model_uri = f'models:/{new_model_name}/{new_model_version}'
    new_model = mlflow.sklearn.load_model(new_model_uri)

    return new_model

vectorizer = pickle.load(open('models/vectorizer.pkl','rb'))

#model = load_model()
model = pickle.load(open('models/model.pkl','rb'))


@app.get('/')
def home():
    return "Hello"

@app.post('/predict')
def predict(data:UserInput):
    text = data.user_query

    # clean
    text = normalize_text(text)

    # bow
    features = vectorizer.transform([text])

    # Convert sparse matrix to DataFrame
    features_df = pd.DataFrame.sparse.from_spmatrix(features)
    features_df = pd.DataFrame(features.toarray(), columns=[str(i) for i in range(features.shape[1])])

    # prediction
    result = model.predict(features_df)

    return {"sentiment":"sadness" if str(result[0]) == "0" else "happiness"}

