from connection import get_connection
from sqlalchemy import text


def get_table(table):
    db = get_connection()
    return db.execute(text(f"select * from {table}"))


def get_specifics(columns, table):
    db = get_connection()
    string = ", ".join(columns)
    return db.execute(text(f"select {string} from {table}"))
