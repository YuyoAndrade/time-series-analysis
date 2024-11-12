FROM python:3.11.10

RUN apt update && apt upgrade -y 

COPY .env /.env
# COPY daily.csv /streamlit/daily.csv

COPY ../requirements.txt /requirements.txt

RUN pip install --no-cache-dir -r /requirements.txt

COPY /streamlit /streamlit

COPY ../database /streamlit/database
COPY ../models /streamlit/models
