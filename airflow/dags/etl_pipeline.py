import pyodbc
import os
import logging
import pandas as pd

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from dotenv import load_dotenv

import sys
from pathlib import Path

project_root = Path().resolve()
sys.path.append(str(project_root))

from database.utils import get_table

load_dotenv(override=True)

DATABASE_STRING = os.getenv("DATABASE_STRING")

DAILY_TABLE = "iar_ocupaciones.dbo.daily"


default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 1, 1),
}

with DAG(
    "etl_pipeline",
    description="DAG for Extracting, Transforming, and Loading data to Azure.",
    default_args=default_args,
    schedule_interval="@monthly",
    catchup=False,
) as dag:

    def extract():
        table = "iar_ocupaciones"
        parquet_path = "./database/tmp/etl.parquet"

        logging.info(f"Extracting table {table}...")

        result = get_table(table=table)
        df = pd.DataFrame(result.fetchall(), columns=result.keys())

        logging.info(f"Table {table} extracted.")
        logging.info(f"Amounts of rows: {len(df)}")
        logging.info(f"Saving table in {parquet_path}...")

        df.to_parquet(parquet_path)

        logging.info(f"Table saved in {parquet_path}...")

        return parquet_path

    def transform(**context):
        path = context["ti"].xcom_pull(task_ids="extract")

        logging.info(f"Reading table from {path}")

        df = pd.read_parquet(path)
        df = df.drop_duplicates()

        logging.info("Tranforming to daily table...")

        df["Fecha_hoy"] = pd.to_datetime(df["Fecha_hoy"])
        df["daily"] = df["Fecha_hoy"].dt.strftime("%Y-%m-%d")

        df = df.drop("Fecha_hoy", axis=1)
        df = df[["daily", "ing_hab"]]
        df = df.groupby(by="daily").sum()
        df = df.sort_values(by="daily", axis=0, ascending=True)

        logging.info("Daily table transformed.")

        df.to_parquet(path)
        return path

    def load(**context):
        path = context["ti"].xcom_pull(task_ids="extract")
        df = pd.read_parquet(path)
        cnxn = pyodbc.connect(DATABASE_STRING)
        cursor = cnxn.cursor()
        # Insert Dataframe into SQL Server:
        rows = len(df)
        i = 1
        for row in df.itertuples():
            logging.info(f"Row: {i} of {rows} rows...")
            cursor.execute(
                f"""
                INSERT INTO {DAILY_TABLE} (Day,ing_hab) 
                select ?, ? 
                where not exists (
                    select 1
                    from {DAILY_TABLE}
                    where Day = ?
                )
                """,
                (row[0], row[1], row[0]),
            )
            i += 1

        logging.info("Committing...")

        cnxn.commit()
        cursor.close()

        logging.info("Data uploaded to Azure")
        logging.info(df.head().to_string())
        return True

    extracting = PythonOperator(task_id="extract", python_callable=extract)

    transforming = PythonOperator(task_id="transform", python_callable=transform)

    loading = PythonOperator(task_id="load", python_callable=load)

    extracting >> transforming >> loading
