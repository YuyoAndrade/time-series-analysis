import os
import logging
from sqlalchemy import create_engine
from dotenv import load_dotenv

from sqlalchemy.orm import sessionmaker
from azure.storage.blob import BlobServiceClient

load_dotenv(override=True)

db_user = os.getenv("DATABASE_USER")
db_password = os.getenv("DATABASE_PASSWORD")
db_name = os.getenv("DATABASE_NAME")
db_port = os.getenv("DATABASE_PORT")
db_server = os.getenv("DATABASE_HOST")
blob_connection_string = os.getenv("BLOB_STRING")
database_connection_string = os.getenv("DATABASE_STRING")

# Define de DATABASE URL.
SQLALCHEMY_DATABASE_URL = (
    f"sqlserver://{db_user}:{db_password}@{db_server}:{db_port}/{db_name}"
)

SQLSERVER_DATABASE_URL = f"mssql+pyodbc:///?odbc_connect={database_connection_string}"

engine = create_engine(SQLSERVER_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_connection():
    logging.info("Connecting to Azure SQL Database.")
    db = SessionLocal()
    try:
        logging.info("Connected...")
        return db
    except Exception as ex:
        logging.info("Error: " + str(ex))
    finally:
        db.close()


def create_blob_client():
    try:
        logging.info("Connecting to Azure Blob Storage.")
        blob_source_service_client = BlobServiceClient.from_connection_string(
            blob_connection_string
        )
        logging.info("Connected...")
        return blob_source_service_client

    except Exception as ex:
        logging.info("Error: " + str(ex))
