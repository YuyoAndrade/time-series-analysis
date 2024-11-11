from .connection import get_connection
from sqlalchemy import text
import pandas as pd
import os
import logging
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError, AzureError


def download_blobs_to_file(
    blob_service_client: BlobServiceClient, container_name, download_path
):
    try:
        # Get the container client
        container_client = blob_service_client.get_container_client(
            container=container_name
        )

        # Check if the container exists
        if not container_client.exists():
            raise ResourceNotFoundError(
                f"The container '{container_name}' does not exist."
            )

        # List all blobs in the container
        blobs = container_client.list_blobs()

        logging.info(
            f"Starting download of blobs from container '{container_name}' to '{download_path}'..."
        )

        for blob in blobs:
            blob_client = container_client.get_blob_client(blob)

            # Construct the full local path for the blob
            # This preserves the blob's directory structure
            local_file_path = os.path.join(download_path, blob.name)

            # Create local directories if they do not exist
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

            logging.info(f"Downloading blob: {blob.name}...")

            # Download the blob's content to the local file
            with open(local_file_path, "wb") as download_file:
                download_stream = blob_client.download_blob()
                download_file.write(download_stream.readall())

            logging.info(
                f"Successfully downloaded '{blob.name}' to '{local_file_path}'."
            )

        logging.info("All blobs have been downloaded successfully.")

    except ResourceNotFoundError as e:
        logging.info(f"Error: {e.message}")
    except AzureError as e:
        logging.info(f"An Azure error occurred: {e.message}")
    except Exception as e:
        logging.info(f"An unexpected error occurred: {str(e)}")


def upload_blob_file(blob_service_client: BlobServiceClient, container_name, file):
    container_client = blob_service_client.get_container_client(
        container=container_name
    )
    logging.info(f"Uploading model {file}...")
    with open(file=file, mode="rb") as data:
        blob_client = container_client.upload_blob(name=file, data=data, overwrite=True)
    logging.info(f"Model {file} - Uploaded")


def get_table(table):
    db = get_connection()
    return db.execute(text(f"select * from {table}"))


def get_specifics(columns, table):
    db = get_connection()
    columns = [f'{table}."{c}"' for c in columns]
    statement = ", ".join(columns)
    return db.execute(text(f"select {statement} from {table} where moneda_cve = 2"))


def create_weekly_dataframe(columns, table):
    result = get_specifics(columns=columns, table=table)

    df = pd.DataFrame(result.fetchall(), columns=result.keys())
    df = df.drop_duplicates()

    df["Fecha_hoy"] = pd.to_datetime(df["Fecha_hoy"])
    df["weekly"] = df["Fecha_hoy"].dt.strftime("%Y-%W")

    df = df.drop("Fecha_hoy", axis=1)
    df = df.groupby(by="weekly").sum()

    return df.sort_values(by="weekly", axis=0, ascending=True)


def create_daily_dataframe(columns, table):
    result = get_specifics(columns=columns, table=table)

    df = pd.DataFrame(result.fetchall(), columns=result.keys())
    df = df.drop_duplicates()

    df["Fecha_hoy"] = pd.to_datetime(df["Fecha_hoy"])
    df["weekly"] = df["Fecha_hoy"].dt.strftime("%Y-%D")

    df = df.drop("Fecha_hoy", axis=1)
    df = df.groupby(by="weekly").sum()

    return df.sort_values(by="weekly", axis=0, ascending=True)
