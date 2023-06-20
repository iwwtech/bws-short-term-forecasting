import os

### Orion
ORION_CONTAINER_NAME = os.getenv("ORION_CONTAINER_NAME")
ORION_PORT = os.getenv("ORION_PORT")
ORION_BASE_URL = f"http://{ORION_CONTAINER_NAME}:{ORION_PORT}/v2"

### Core Tool
CORE_PORT = os.getenv("CORE_PORT")
CORE_BASE_URL = f"http://core-tool:{CORE_PORT}"
FIWARE_ENABLED = os.getenv("FIWARE_ENABLED", "False").lower() in ("true", "1", "t")

### Database
# NOTE: MONGO_CONTAINER_NAME is not defined in the .env file, but in the
# docker-compose.yml file for consistency with the container naming
MONGO_HOST = os.getenv("MONGO_CONTAINER_NAME")
MONGO_USER = os.getenv("MONGO_ROOT_USERNAME")
MONGO_PASS = os.getenv("MONGO_ROOT_PASSWORD")
DB_NAME = os.getenv("MONGO_DATABASE")

PM_COLCTN = os.getenv("PHYSICAL_METER_COLLECTION", "devices")
VM_COLCTN = os.getenv("VIRTUAL_METER_COLLECTION", "virtualDevices")
PM_MEAS_COLCTN = os.getenv(
    "PHYSICAL_METER_MEASUREMENT_COLLECTION", "deviceMeasurements"
)
MODEL_COLCTN = os.getenv("ML_MODEL_COLLECTION", "mlModels")
STATS_COLCTN = os.getenv("STATISTICS_COLLECTION", "statistics")


### Locality for holiday features
COUNTRY_CODE = os.getenv("COUNTRY_CODE", "DE")
SUBDIVISION_CODE = os.getenv("SUBDIVISION_CODE", "")

### Models
DEFAULT_ALGORITHM = os.getenv("DEFAULT_ALGORITHM", "xgboost")
DEFAULT_MEASUREMENT_UNIT = os.getenv("DEFAULT_MEASUREMENT_UNIT", None)

# Whether external features from the weather agent should be used
USE_WEATHER = os.getenv("USE_WEATHER", "False").lower() in ("true", "1", "t")

MONGO_DB_NAME = os.getenv("MONGO_DATABASE")
MODEL_FOLDER = os.path.join("app", "data", "models")
os.makedirs(MODEL_FOLDER, exist_ok=True)

N_EPOCHS_TORCH = int(os.getenv("N_EPOCHS_TORCH", 100))

### Hyperparameter optimization with Ray Tune (BOHB)
RAY_N_MODELS = int(os.getenv("RAY_N_MODELS", 32))
RAY_MAX_CONCURRENT = int(os.getenv("RAY_MAX_CONCURRENT", 4))
RAY_REDUCTION_FACTOR = int(os.getenv("RAY_REDUCTION_FACTOR", 2))
RAY_NUM_CPUS = int(os.getenv("RAY_NUM_CPUS", 8))

# Note that the number of GPUs is limited to 1 in the current implementation
RAY_NUM_GPUS = min(1, int(os.getenv("RAY_NUM_GPUS", 0)))
RAY_NUM_GPUS_PER_TRIAL = float(os.getenv("RAY_NUM_GPUS_PER_TRIAL", 0.5))
RAY_MAX_T = int(os.getenv("RAY_MAX_T", 100))
