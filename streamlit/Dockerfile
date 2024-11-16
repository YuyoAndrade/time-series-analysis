FROM python:3.11.10

RUN apt update && apt upgrade -y 
RUN apt install -y curl
#Debian 12
RUN curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg
RUN curl https://packages.microsoft.com/config/debian/12/prod.list | tee /etc/apt/sources.list.d/mssql-release.list

RUN apt-get update
RUN ACCEPT_EULA=Y apt-get install -y msodbcsql18

COPY .env /.env
# COPY daily.csv /streamlit/daily.csv

COPY ../requirements.txt /requirements.txt

RUN pip install --no-cache-dir -r /requirements.txt

COPY /streamlit /streamlit

COPY ../database /streamlit/database
COPY ../models /streamlit/models
