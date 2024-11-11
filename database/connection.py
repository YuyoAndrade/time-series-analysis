import os
import logging
from sqlalchemy import create_engine
from dotenv import load_dotenv

# from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from azure.storage.blob import BlobServiceClient

load_dotenv(override=True)

db_user = os.getenv("DATABASE_USER")
db_password = os.getenv("DATABASE_PASSWORD")
db_name = os.getenv("DATABASE_NAME")
db_port = os.getenv("DATABASE_PORT")
db_server = os.getenv("DATABASE_HOST")
blob_connection_string = os.getenv("BLOB_STRING")

# Define de DATABASE URL.
SQLALCHEMY_DATABASE_URL = (
    f"postgresql://{db_user}:{db_password}@{db_server}:{db_port}/{db_name}"
)
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_connection():
    db = SessionLocal()
    try:
        return db
    finally:
        db.close()


def create_blob_client():
    try:
        blob_source_service_client = BlobServiceClient.from_connection_string(
            blob_connection_string
        )
        logging.info("Connection String -- Connected.")
        return blob_source_service_client

    except Exception as ex:
        logging.info("Error: " + str(ex))
