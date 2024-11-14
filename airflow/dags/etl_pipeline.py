import sys
from pathlib import Path

project_root = Path().resolve()
sys.path.append(str(project_root))

from database.connection import get_connection
from database.utils import *
from sqlalchemy import insert
from airflow import DAG
from airflow.operators.python import PythonOperator
import logging
import pandas as pd
from datetime import datetime


dataset = pd.read_csv('iar_Ocupaciones.csv') 

def extract():
    try:
        df = dataset  # Ensure correct file path
        logging.info(f"Data extracted: \n{df.head()}")
        df.to_parquet("extract.parquet")
        return "extract.parquet"  # Return path for use in XCom
    except Exception as e:
        logging.error(f"Error in extract function: {e}")
        raise

    '''
    df = pd.read_csv('iar_Ocupaciones.csv')
    #tabla = get_table(table) # toma la tabla
    #df = create_dataframe(columns, tabla) # crea un dataframe en base a la tabla
    logging.info(df.to_string())
    df.to_parquet("extract.parquet")
    return "extract.parquet"  #!!!!formato binario!!!!! instalar parquet
    '''


def transform(**kwargs):
    try:
        ti = kwargs["ti"]
        #filepath = ti.xcom_pull(task_ids="extract_task")
        filepath = 'extract.parquet'
        df = pd.read_parquet(filepath, engine="pyarrow")
        # df = create_daily_dataframe(columns=["Fecha_hoy", "ing_hab"], table="iar_ocupaciones")

        df["Fecha_hoy"] = pd.to_datetime(df["Fecha_hoy"])
        df["daily"] = df["Fecha_hoy"].dt.strftime("%Y-%m-%d")
        df = df.drop("Fecha_hoy", axis=1)
        df = df.groupby(by="daily").sum()
        #df.sort_values(by="daily", axis=0, ascending=True)

        df.to_parquet("transform.parquet")
        logging.info("Data transformed and saved.")
        return "transform.parquet"
    except Exception as e:
        logging.error(f"Error in transform function: {e}")
        raise


    '''
    ti = kwargs["ti"]
    filepath = ti.xcom_pull(task_ids="extract_task")
    df = pd.read_parquet(filepath)
    df = change_date_format(df) # transforma el formato de fecha
    #df.to_parquet("transform.parquet")
    return "transform.parquet"
    '''


def load(**kwargs):
    try:
        ti = kwargs["ti"]
        filepath = 'transform.parquet'
        #filepath = ti.xcom_pull(task_ids="transform_task")
        df = pd.read_parquet(filepath)

        #db = get_Connection()  # Ensure this function returns a valid connection
        #data = df.to_dict(orient="records")
        #stmt = insert(table).values(data)  # Make sure `table` variable is valid here
        #db.execute(stmt)
        #db.commit()
        logging.info("Data loaded to database.")
        logging.info(df.head().to_string())
    except Exception as e:
        logging.error(f"Error in load function: {e}")
        raise


    '''
    ti = kwargs["ti"]
    filepath = ti.xcom_pull(task_ids="transform_task")
    df = pd.read_parquet(filepath)

    # db = getConnection()
    # data = df.to_dict(orient="records") # convierte de dataframe a dict para poder añadir los valores

    # stmt = insert(df) # inserta en el dataframe
    # db.execute(stmt, data) # los valores de data en el stmt
    # db.commit()
    logging.info(df.to_string())
    '''

default_args = {
    "owner": "airflow",
    "start_date": datetime(2024,1,1),
}

with DAG(
    dag_id="etl_pipeline", default_args=default_args, schedule_interval="@daily", catchup=False
) as dag:
    # Define your tasks here
    extract_task = PythonOperator(
        task_id="extract_task",
        python_callable=extract,
        op_kwargs={"columns": ["column1", "column2"], "table": "table_name"},
    )

    transform_task = PythonOperator(
        task_id="transform_task",
        python_callable=transform,
        provide_context=True,
    )

    load_task = PythonOperator(
        task_id="load_task",
        python_callable=load,
        provide_context=True,
    )

    # Set up the task dependencies
    extract_task >> transform_task >> load_task
