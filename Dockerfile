# Use the official Airflow image as the base
FROM apache/airflow:2.10.2

USER root
RUN apt update && apt upgrade -y

USER airflow
# Copy the requirements.txt file into the container
COPY requirements.txt /requirements.txt

# Install the required Python packages
RUN pip install --no-cache-dir -r /requirements.txt
