from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import logging

default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 1, 1),
}

with DAG(
    "model_pipeline",
    description="DAG for ML models creation, testing, load.",
    default_args=default_args,
    schedule_interval="@daily",
    catchup=False,
) as dag:

    def extract_models():
        logging.info("Extracting...")
        pass

    def train():
        logging.info("Training...")
        pass

    def test():
        logging.info("Testing...")
        pass

    def load():
        logging.info("Loading...")
        pass

    extracting = PythonOperator(
        task_id="extract_models", python_callable=extract_models
    )

    training = PythonOperator(task_id="train", python_callable=train)

    testing = PythonOperator(task_id="test", python_callable=test)

    loading = PythonOperator(task_id="load", python_callable=load)

    extracting >> training >> testing >> loading
