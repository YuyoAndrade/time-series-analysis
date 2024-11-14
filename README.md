# **MLOps for Time Series Modeling**

> A complete MLOps pipeline for building, training, and deploying time series models with integrated ETL processes and a user-friendly Streamlit interface.

---

## **Table of Contents**
1. [Features](#features)
2. [Installation](#installation)
3. [Usage](#usage)

---

## **Features**

### **Time Series Models**
This project includes specialized time series models:
- **LSTM** (Long Short-Term Memory)
- **Prophet**
- **Holt-Winters**

### **Apache Airflow Pipelines**
1. **ETL Pipeline**: 
   - **Extract**: Retrieves data from the client's database.
   - **Transform**: 
     - Groups data by day and calculates total revenue.
     - Saves intermediate transformations in a `.parquet` file at each step to ensure data consistency and facilitate debugging.
   - **Load**: Writes the transformed data into Azure SQL Server.

2. **Model Pipeline**:
   - **Create Models**: Reads model configurations from `to_create.py` (via `MODELS_TO_CREATE` dictionary) and generates `.pkl` files for each model.
   - **Training**: Trains each model using its specific method.
   - **Testing**: Tests the model, evaluates performance, and saves metrics.
   - **Loading**: Updates `.pkl` files with the latest model structure and uploads them to Azure Blob Storage.
   - **Integration with MLflow**: Tracks training and testing results and saves metrics to Azure ML.

### **Streamlit Interface**
- Visualizes:
  - Model explanations
  - Test results and metrics
  - Prediction capabilities
- Provides a user-friendly interface for client interaction.

### **Dockerized Deployment**
- Containers for:
  - Apache Airflow (Postgres, Redis, Init, Webserver, Scheduler, Worker, Triggerer)
  - Streamlit
- Simplified deployment with Docker Compose.

---

## **Installation**

### **Prerequisites**
- **Docker** and **Docker Compose** installed on your machine.
- An **Azure Account** with the following services configured:
  - **Azure SQL Database**: For storing transformed data.
  - **SQL Server**: To connect and interact with the Azure SQL Database.
  - **Azure Blob Storage**: For saving model artifacts (`.pkl` files).
  - **Azure Machine Learning**: For managing and tracking model training and testing.
- Required environment variables configured in a `.env` file.

#### **Environment Variables**
Create a `.env` file in the root of the project with the following variables:

1. **Airflow Configuration**:
   ```plaintext
    AIRFLOW_UID = 1000 (Linux) or 50000 (Mac or Windows)
    AIRFLOW_DB = "airflow_db"
    AIRFLOW_DB_USER = <your airflow username>
    AIRFLOW_DB_PASSWORD = <your airflow password>
    AIRFLOW_DB_HOST = <your airflow host>
    AIRFLOW_DB_PORT = <your airflow port>
    AIRFLOW_IMAGE_NAME = "apache/airflow:2.10.2-python3.11"
    AIRFLOW_SERVER = {your airflow server}
    AIRFLOW_PROJ_DIR = "./airflow"
   ```

2. **Azure SQL Database**:
   ```plaintext
    DATABASE_USER = <your username>
    DATABASE_PASSWORD = <your password>
    DATABASE_STRING = Format: Driver={<your sql server driver>};Server={<your sql server>};Database={<your database name>};Uid={username};Pwd={password};Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"
   ```

3. **Azure Blob Storage**:
   ```plaintext
    BLOB_STRING = <your connection string> (obtain it by going to your azure blob storage -> Security + Network -> Shared access signature -> Generate SAS and connection string)
    BLOB_MODEL_CONTAINER = <container name to store models>
   ```

4. **Models Configuration**:
   ```plaintext
    LSTM_MODEL = <name to save LSTM model.>
    HOLT_WINTERS_MODEL = <name to save Holt-Winters model.>
    PROPHET = <name to save Prophet model.>
   ```

### **Setup**
1. Clone the repository:
   ```bash
   git clone https://github.com/YuyoAndrade/time-series-analysis.git
   cd time-series-analysis
   ```

2. Add your `.env` file to the root directory.

3. Build and initialize Docker containers:
   ```bash
   docker compose up airflow-init --build
   docker compose up --build
   ```

4. Access the services:
   - **Airflow**: [http://localhost:8080](http://localhost:8080)
      - username: airflow
      - password: airflow
   - **Streamlit**: [http://localhost:8501](http://localhost:8501)

---

## **Usage**

### **Running the Pipelines**
1. **ETL Pipeline**:
   - **Extracts** data from the client’s database.
   - **Transform**:
     - Groups the data by day.
     - Sums daily revenue.
   - **Loads** the final transformed data into Azure SQL Server.
   - At each step, saves or updates the data in `.parquet` files for consistency and traceability.

2. **Model Pipeline**:
   - Executes the following tasks:
     1. **Create Models**: Generates `.pkl` files with model structures from `MODELS_TO_CREATE` in `to_create.py`.
     2. **Training**: Trains the models using their respective algorithms.
     3. **Testing**: Evaluates the models and updates metrics in `.pkl` files.
     4. **Loading**: Uploads the finalized `.pkl` files to Azure Blob Storage.

3. MLflow is integrated at each step to track the models’ performance and store results in Azure ML.

### **Streamlit Interface**
- Open the Streamlit app to:
  - View model explanations, test results, and metrics.
  - Use the prediction functionality to make forecasts.

---
