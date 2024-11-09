from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import logging
import pickle
import sys
from pathlib import Path

project_root = Path().resolve()
sys.path.append(str(project_root))
from models.neuralnetworks import LSTM

train = 0.65
validation = 0.15
test = 0.2

models_to_create = ["LSTM"]

default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 1, 1),
}

with DAG(
    "model_pipeline",
    description="DAG for ML models creation, testing, load.",
    default_args=default_args,
    schedule_interval="@monthly",
    catchup=False,
) as dag:

    def create_models():
        models = []
        for m in models_to_create:
            logging.info(f"Creating model {m}...")
            model = LSTM(
                name="prueba",
                created_at="2024-11-03",
                version="1.0",
                metrics=[],
                length=2,
            )
            model_path = f"/tmp/{m}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            models.append(model_path)
        return models

    def training(**context):
        logging.info("Training...")
        models = context["ti"].xcom_pull(task_ids="create")
        logging.info(f"Models to train {models}")
        for model_path in models:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            logging.info(f"Training model {model.name}...")
        return models

    def testing(**context):
        logging.info("Testing...")
        # models = context["ti"].xcom_pull(task_ids="training")
        # for model in models:
        #     logging.info(f"Testing model {model.name}...")
        # return models

    def loading(**context):
        logging.info("Loading...")
        # models = context["ti"].xcom_pull(task_ids="training")
        # for model in models:
        #     logging.info(f"Loading model {model.name}...")
        # pass

    creating = PythonOperator(task_id="create", python_callable=create_models)

    training = PythonOperator(task_id="train", python_callable=training)

    testing = PythonOperator(task_id="test", python_callable=testing)

    loading = PythonOperator(task_id="load", python_callable=loading)

    creating >> training >> testing >> loading
