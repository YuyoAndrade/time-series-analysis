# Official Airflow image
FROM apache/airflow:2.10.2

USER root
RUN apt update && apt upgrade -y

COPY .env /opt/airflow/.env
# COPY daily.csv /opt/airflow/daily.csv

USER airflow

COPY requirements.txt /requirements.txt
COPY iar_Ocupaciones.csv /opt/airflow/iar_Ocupaciones.csv

RUN pip install --no-cache-dir -r /requirements.txt
