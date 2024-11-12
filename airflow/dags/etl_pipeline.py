from ..database.connection import get_connection
from ..database.utils import *
from sqlalchemy import insert
from airflow import DAG
from airflow.operators.python import PythonOperator
import logging


def extract(columns, table, **kwargs):
    # tabla = get_table(table) # toma la tabla
    # df = create_dataframe(columns, tabla) # crea un dataframe en base a la tabla
    df = pd.DataFrame({"x": 2.5})
    logging.info(df.to_string())
    df.to_parquet("extract.parquet")
    return "extract.parquet"  #!!!!formato binario!!!!! instalar parquet


def transform(**kwargs):
    ti = kwargs["ti"]
    filepath = ti.xcom_pull(task_ids="extract_task")
    df = pd.read_parquet(filepath)
    df = df.append({"x": 3.5})
    # df = pd.DataFrame.from_dict(filepath)
    # df = change_date_format(df) # transforma el formato de fecha
    df.to_parquet("transform.parquet")
    return "transform.parquet"


def load(**kwargs):
    ti = kwargs["ti"]
    filepath = ti.xcom_pull(task_ids="transform_task")
    df = pd.read_parquet(filepath)

    # db = getConnection()
    # data = df.to_dict(orient="records") # convierte de dataframe a dict para poder añadir los valores

    # stmt = insert(df) # inserta en el dataframe
    # db.execute(stmt, data) # los valores de data en el stmt
    # db.commit()
    logging.info(df.to_string())


tabla = extract()
df = transform(tabla)
load(df)


with DAG(
    dag_id="etc_pipeline", default_args=default_args, schedule_interval="@daily"
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
