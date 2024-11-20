import mlflow
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.to_create import MODELS_TO_CREATE
import pandas as pd
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Model, ManagedOnlineEndpoint, ManagedOnlineDeployment
from azure.ai.ml.constants import AssetTypes
import uuid

# Cambiamos el directorio de regreso al folder
azure_folder = os.path.dirname(__file__)
os.chdir(azure_folder)

# MLflow va documentar el entrenamiento del modelo
mlflow.autolog()

# 1. Determinamos un directorio local y creamos la carpeta para guardar el modelo
local_container_path = os.getcwd()
download_path = os.path.join(local_container_path, "downloaded_model")
os.makedirs(download_path, exist_ok=True)

# 2. Creamos el LSTM
modelo_lstm = MODELS_TO_CREATE["LSTM"]
modelo_lstm.build_model()

# 3. Importamos los datos para el modelo
df = pd.read_csv("daily.csv")

# 4. Entrenamos el modelo usando MLflow
status, run_id = modelo_lstm.train(df, 0.65, 0.15)
print(f"The run id is: {run_id}")

# 5. Descargamos el modelo que acabamos de registrar en MLflow localmente 
model_uri = f"runs:/{run_id}/model"
mlflow.artifacts.download_artifacts(run_id=run_id, dst_path=download_path)

# 6. Configuramos le cliente de azure ml
credential = DefaultAzureCredential()
ml_client = MLClient(
    credential=credential,
    subscription_id="e55be53a-84c8-425d-8430-8151a50b6e6d",
    resource_group_name="TCAML",
    workspace_name="tcaml-workspace",
)

# 7. Registrar el modelo en Azure ML
model_name = "LSTM_Amigos"
model_description = "Modelo LSTM para predicciones de series de tiempo"

model = Model(
    path=os.path.join(download_path, "model"),  # Path to the downloaded model directory
    type=AssetTypes.MLFLOW_MODEL,                  # Specify the model type as MLflow model
    name=model_name,
    description=model_description,
)

registered_model = ml_client.models.create_or_update(model)
print(f"Model registered with name: {registered_model.name} and version: {registered_model.version}")

# 8. Creacion del endpoint en azure
online_endpoint_name = "lstm-endpoint-" + str(uuid.uuid4())[:8]
endpoint = ManagedOnlineEndpoint(
    name=online_endpoint_name,
    description="This is the endpoint for the LSTM model",
    auth_mode="key",
    tags={
        "training_dataset": "daily",
    },
)

endpoint = ml_client.online_endpoints.begin_create_or_update(endpoint).result()
endpoint = ml_client.online_endpoints.get(name=online_endpoint_name)
print(f'Endpoint "{endpoint.name}" with provisioning state "{endpoint.provisioning_state}" is retrieved')

# 9. Create the deployment
deployment_name = "lstm-deployment-amigos"
registered_model_name = model_name
latest_model_version = max([int(m.version) for m in ml_client.models.list(name=registered_model_name)])
model_deploy = ml_client.models.get(name=registered_model_name, version=latest_model_version) # Agarramos el modelo más reciente

deployment = ManagedOnlineDeployment(
    name=deployment_name,
    endpoint_name=online_endpoint_name,
    model=model_deploy,
    instance_type="Standard_DS2_v2",  # Instancia para el deployment # Standard_DS3_v2
    instance_count=1,
    app_insights_enabled=True
)

ml_client.online_deployments.begin_create_or_update(deployment).result()

# Enrutamos el 100% del trafico al deployment
endpoint.traffic = {deployment_name: 100}
ml_client.online_endpoints.begin_create_or_update(endpoint).result()
print(f"Deployment '{deployment_name}' is live at endpoint '{online_endpoint_name}'.")
