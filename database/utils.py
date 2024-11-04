from .connection import get_connection
from sqlalchemy import text
import pandas as pd


def get_table(table):
    db = get_connection()
    return db.execute(text(f"select * from {table}"))


def get_specifics(columns, table):
    db = get_connection()
    columns = [f'{table}."{c}"' for c in columns]
    statement = ", ".join(columns)
    return db.execute(text(f"select {statement} from {table} where moneda_cve = 2"))


def create_dataframe(columns, table):
    result = get_specifics(columns=columns, table=table)

    df = pd.DataFrame(result.fetchall(), columns=result.keys())
    df = df.drop_duplicates()

    df["Fecha_hoy"] = pd.to_datetime(df["Fecha_hoy"])
    df["weekly"] = df["Fecha_hoy"].dt.strftime("%Y-%W")

    df = df.drop("Fecha_hoy", axis=1)
    df = df.groupby(by="weekly").sum()

    return df.sort_values(by="weekly", axis=0, ascending=True)
