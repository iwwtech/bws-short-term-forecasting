import os
import re
import joblib
import pandas as pd
from math import ceil

from app.utils.exception_handling import (
    BadRequestException,
    DataProcessingException,
    DeleteResourceException,
    MethodNotAllowedException,
    ResourceNotFoundException,
    SaveResourceException,
)

from app.utils.feature_processing import get_start_of_day

from app.schemas import (
    VirtualMeter as VirtualMeterSchema,
    TrainArguments as TrainArgumentsSchema,
)
from app.utils.constants import (
    MODEL_FOLDER,
    DEFAULT_ALGORITHM,
    DEFAULT_MEASUREMENT_UNIT,
    DB_NAME,
    MONGO_HOST,
    MONGO_USER,
    MONGO_PASS,
    VM_COLCTN,
    PM_COLCTN,
    PM_MEAS_COLCTN,
    MODEL_COLCTN,
    STATS_COLCTN,
)
from marshmallow import EXCLUDE
from datetime import datetime, timedelta

from torch import save as torch_save, load as torch_load

from pymongo import MongoClient

_client = None


def get_mongo_client():
    global _client
    if _client is None:
        _client = MongoClient(
            f"mongodb://{MONGO_USER}:{MONGO_PASS}@{MONGO_HOST}:27017/"
        )
    return _client


def raise_if_not_exists(_id: str, collection: str, name: str):
    """
    Raises a ResourceNotFoundException if no object with the given id could be found.
    :param _id: The id of the object to look for.
    :param collection: The MongoDB collection to search in.
    :param name: The name of the object in order to create a meaningful
                 response message. For example "Virtual Meter".
    """
    client = get_mongo_client()
    if client[DB_NAME][collection].find_one({"_id": _id}) is None:
        raise ResourceNotFoundException(name + " with id " + _id + " not found.")


def raise_if_meter_not_exists(meter_id: str):
    """
    Raises a ResourceNotFoundException if no physical or virtual meter
    with the given id can be found.
    """
    if is_virtual(meter_id):
        raise_if_not_exists(meter_id, VM_COLCTN, "Virtual Meter")
    else:
        raise_if_not_exists(meter_id, PM_COLCTN, "Physical Meter")


def raise_if_model_not_exists(meter_id: str, algorithm: str):
    """
    Raises a ResourceNotFoundException if no model can be found
    for the given meter and algorithm
    """
    client = get_mongo_client()
    # Note: The option "i" makes the regex case insensitive
    model = client[DB_NAME][MODEL_COLCTN].find_one(
        {"refMeter": meter_id, "algorithm": {"$regex": algorithm, "$options": "i"}}
    )
    if model is None:
        raise ResourceNotFoundException(
            f'No model found for meter "{meter_id}" and algorithm "{algorithm}".'
        )


def get_physical_meters(pmid: str = None, include_measurements: bool = False):
    """
    Returns the meta data of a physical meter (device).
    :param pmid: The physical meter id that identifies the pm to retrieve.
                 If pmid is specified, the measurements are returned as well.
    :param include_measurements: If True, the measurements associated with the meter
                                 are returned as well.
    """
    client = get_mongo_client()
    is_single = pmid is not None
    if is_single:
        raise_if_not_exists(pmid, PM_COLCTN, "Physical Meter")

    # Retrieve meta data of physical meters / devices
    pmids = [pmid] if pmid else list(client[DB_NAME][PM_COLCTN].distinct("_id"))
    query = {"_id": {"$in": pmids}}
    mask = {"_id": 0}
    meta = list(client[DB_NAME][PM_COLCTN].find(query, mask))

    result = {"meter": meta} if is_single else {"meters": meta}
    if include_measurements:
        measurements = [get_pm_measurements(pm["id"], compact=True) for pm in meta]
        result["measurements"] = measurements
    return result


def get_measurements(
    meter_id: str,
    aggregate: bool = True,
    diff: bool = False,
    max_missing: float = 0.20,
):
    """
    Returns the measurements of a meter as a dataframe.
    :param meter_id: The meter id that identifies the meter to retrieve.
    :param aggregate: If True, the measurements of all submeters are aggregated into one time series.
    :param diff: If True, the difference between the measurements is returned instead of the absolute values.
                 The difference indicates the change between two consecutive measurements.
    :param max_missing: The maximum percentage of missing values in the time series allowed.
    """
    if is_virtual(meter_id):
        df_measurements = get_vm_measurements(meter_id)
    else:
        df_measurements = get_pm_measurements(meter_id, as_df=True)

    # Check how many values are missing
    missing = df_measurements.isna().sum().sum()
    percentage_missing = missing / df_measurements.size
    if percentage_missing > max_missing:
        raise DataProcessingException(
            f"Too many missing values when combining measurements for meter with id {meter_id}."
            + f"Missing: {percentage_missing*100}%"
        )
    else:
        df_measurements = df_measurements.interpolate(method="linear")

    if aggregate and is_virtual(meter_id):
        df_measurements = df_measurements.sum(axis=1).to_frame(name="value")
    if diff:
        # Calculate the difference between two consecutive measurements
        # and set the first row to 0, as it becomes NaN after the computation
        df_measurements = df_measurements.diff()
        df_measurements.iloc[0] = 0

    return df_measurements


def get_vm_measurements(vmid):
    """
    Returns the measurements associated with (the submeters of) a virtual meter as a dataframe.
    :param vmid: The virtual meter id that identifies the vm to retrieve.
    """
    vm = get_virtual_meter(vmid)
    df_all = None

    # Generate dataframe of measurements of all physical meters
    for sub_mid in vm["submeterIds"]:
        if is_virtual(sub_mid):
            df_sub = get_vm_measurements(sub_mid)
        else:
            df_sub = get_pm_measurements(sub_mid, as_df=True)
        df_all = df_sub if df_all is None else pd.concat([df_all, df_sub], axis=1)
    return df_all


def get_pm_measurements(pmid, check_exists=False, as_df=False, compact=False):
    """
    Returns the measurements of a physical meter.
    :param pmid: The physical meter id that identifies the pm to retrieve.
    :param as_df: If True, the measurements are returned as a dataframe.
    :param compact: If True, the measurements are returned as an object
                    that contains common information like the unit and the
                    measurements as a list of small tuples (timestamp, value).
                    This reduces the total size of the response, as not every
                    measurement has to contain the common information like the unit.
    """
    client = get_mongo_client()
    if check_exists:
        raise_if_not_exists(pmid, PM_COLCTN, "Physical Meter")

    query = {"refDevice": pmid}
    mask = {"_id": 0}
    measurements = list(client[DB_NAME][PM_MEAS_COLCTN].find(query, mask))

    n_unique = len(set([m["unit"] for m in measurements]))
    unit = measurements[0]["unit"] if n_unique == 1 else None
    assert (
        n_unique < 2 and unit == DEFAULT_MEASUREMENT_UNIT
    ), "Converting between different measurement units is not supported, expected unit MQH (equivalent to m³)."

    if as_df:
        columns = {pmid: [m["numValue"] for m in measurements]}
        index = pd.to_datetime([m["dateObserved"] for m in measurements])
        index = pd.Index(index, name="date")
        return pd.DataFrame(columns, index=index).sort_index()

    if compact and len(measurements) > 0:
        return {
            "refDevice": pmid,
            "unit": measurements[0]["unit"],
            "measurements": [
                {"dateObserved": m["dateObserved"], "numValue": m["numValue"]}
                for m in measurements
            ],
        }
    return measurements


def get_virtual_meters():
    """
    Returns the meta data of all virtual meters.
    """
    client = get_mongo_client()
    return list(client[DB_NAME][VM_COLCTN].find({}, {"_id": 0}))


def _get_meter(mid, collection, label):
    """
    Returns the meta data of the specified meter.
    :param mid: The meter id that identifies the meter to retrieve.
    :param collection: The MongoDB collection to search for the meter.
    :param label: The label of the meter type when throwing an error.
    """
    raise_if_not_exists(mid, collection, label)
    client = get_mongo_client()
    query = {"_id": mid}
    mask = {"_id": 0}
    return client[DB_NAME][collection].find_one(query, mask)


def get_virtual_meter(vmid: str):
    """
    Returns the meta data of the specified virtual meter.
    :param vmid: The virtual meter id that identifies the vm to retrieve.
    """
    return _get_meter(vmid, VM_COLCTN, "Virtual Meter")


def get_physical_meter(pmid: str):
    """
    Returns the meta data of the specified physical meter.
    :param pmid: The physical meter id that identifies the pm to retrieve.
    """
    return _get_meter(pmid, PM_COLCTN, "Physical Meter")


def get_meter(mid: str):
    """
    Returns the meta data of the specified meter.
    :param mid: The meter id that identifies the meter to retrieve.
    """
    if is_virtual(mid):
        return get_virtual_meter(mid)
    else:
        return get_physical_meter(mid)


def create_virtual_meter_endpoint(vm_details: dict):
    """
    Creates a new virtual meter in the database.
    :param vm_details: Dict with all properties of the virtual meter.
    """
    client = get_mongo_client()

    desc = (
        "Virtual meter that represents a group of smart meters that measure water flow."
    )
    vm_details["description"] = desc

    # Check that every referenced submeter exists
    meters_referenced = vm_details["submeterIds"]
    query = {"_id": {"$in": meters_referenced}}
    mask = {"_id": 1}
    vms_found = [m["_id"] for m in client[DB_NAME][VM_COLCTN].find(query, mask)]
    pms_found = [m["_id"] for m in client[DB_NAME][PM_COLCTN].find(query, mask)]
    meters_found = vms_found + pms_found
    if len(set(meters_found)) != len(set(meters_referenced)):
        missing_vms = [m for m in meters_referenced if m not in meters_found]
        raise BadRequestException(
            "Could not find all referenced submeters. Ids of missing meters: "
            + " ".join(missing_vms)
        )

    # Check that the submeters are disjoint at every level
    raise_if_not_disjoint(meters_referenced)

    # Check that the list of submeters is not empty
    if len(meters_referenced) == 0:
        raise BadRequestException(
            "The list of submeters is empty, "
            + "please pass a list of valid submeters."
        )

    # Update the supermeter ids of the referenced submeters
    query = {"_id": {"$in": meters_referenced}}
    update = {"$addToSet": {"supermeterIds": vm_details["id"]}}
    client[DB_NAME][VM_COLCTN].update_many(query, update)

    # Store the new virtual meter in MongoDB
    vm_details["_id"] = vm_details["id"]
    vm_details["dateCreated"] = datetime.now().isoformat()
    client[DB_NAME][VM_COLCTN].insert_one(vm_details)
    return vm_details["id"]


def generate_vm_id(name: str = None):
    """
    Generates a new virtual meter id that is unique in the db.
    The id is generated from the name if given, otherwise it
    represents the next free number in the db.
    """
    client = get_mongo_client()

    if name:
        # Check if the name contains illegal characters
        if not re.match("^[a-zA-Z0-9_-]+$", name):
            raise BadRequestException(
                "The name of the virtual meter contains illegal characters. "
                + "Only letters, numbers, underscores and hyphens are allowed."
            )

        # Check that the name is not already used
        vmid = "urn:ngsi-ld:virtualMeter:" + name
        vm_found = client[DB_NAME][VM_COLCTN].find_one({"_id": vmid})
        if vm_found:
            raise BadRequestException(
                "A virtual meter with the name "
                + name
                + " already exists. Please choose another name."
            )
        return vmid
    else:
        # Check all ids in the database that are numeric and find the next free number
        vmids = list(client[DB_NAME][VM_COLCTN].distinct("_id"))
        numeric_ids = []
        for vmid in vmids:
            id_str = vmid.split(":")[-1]
            if id_str.isdigit():
                numeric_ids.append(int(id_str))
        vmid = numeric_ids[-1] + 1 if len(numeric_ids) > 0 else 1
        return "urn:ngsi-ld:virtualMeter:" + str(vmid).zfill(5)


def delete_virtual_meter_endpoint(vmid: str, rm_from_super: bool = False):
    """
    Deletes a virtual meter from the database.
    :param vmid: The id of the virtual meter to delete.
    :param rm_from_super: If true, the virtual meter is removed from the all associated
                         supermeters. If false although the virtual meter is part
                         of a supermeter, an exception is raised.
    """
    client = get_mongo_client()
    raise_if_not_exists(vmid, VM_COLCTN, "Virtual Meter")
    vm = client[DB_NAME][VM_COLCTN].find_one({"_id": vmid})

    if len(vm["supermeterIds"]) > 0 and not rm_from_super:
        raise MethodNotAllowedException(
            "The specified Virtual Meter is part of the following supermeters: "
            + " ".join(vm["supermeterIds"])
            + "\n"
            + "If it should be deleted nevertheless, specify the query parameter "
            + "removeFromSuperMeters=true."
        )
    elif len(vm["supermeterIds"]) > 0:
        # Remove the virtual meter from the associated supermeters
        query = {"submeterIds": vmid}
        update = {"$pull": {"submeterIds": vmid}}
        client[DB_NAME][VM_COLCTN].update_many(query, update)

        # Invalidate the models of all associated supermeters
        query = {"refMeter": {"$in": vm["supermeterIds"]}}
        update = {"$set": {"isModelValid": False}}
        client[DB_NAME][MODEL_COLCTN].update_many(query, update)

    # Remove the virtual meter from the associated submeters
    query = {"_id": {"$in": vm["submeterIds"]}}
    update = {"$pull": {"supermeterIds": vmid}}
    client[DB_NAME][VM_COLCTN].update_many(query, update)

    # Remove all associated model meta entries and binaries
    query = {"refMeter": vmid}
    model_metas = list(client[DB_NAME][MODEL_COLCTN].find(query))
    for model_meta in model_metas:
        client[DB_NAME][MODEL_COLCTN].delete_one({"_id": model_meta["_id"]})
        delete_model_binary(model_meta)

    # Remove the virtual meter from the database
    client[DB_NAME][VM_COLCTN].delete_one({"_id": vmid})


def invalidate_models(vmid: str, algorithm: str = None):
    """
    Invalidates the model(s) of a virtual meter.
    :param vmid: The id of the virtual meter.
    :param algorithm: The algorithm that should be invalidated.
                      If none, all models are invalidated.
    """
    client = get_mongo_client()
    if algorithm:
        filter = {"refMeter": vmid}
    else:
        filter = {"refMeter": vmid, "algorithm": {"$regex": algorithm, "$options": "i"}}
    update = {"$set": {"isValid": False}}
    client[DB_NAME][MODEL_COLCTN].update_many(filter, update)


def get_default_algorithm(meter_id: str):
    """
    Returns the algorithm that is set as default for the given meter.
    """
    client = get_mongo_client()
    query = {"refMeter": meter_id, "isDefault": True}
    model_meta = client[DB_NAME][MODEL_COLCTN].find_one(query)
    if model_meta is None:
        return DEFAULT_ALGORITHM
    else:
        return model_meta["algorithm"]


def has_default(meter_id: str):
    """
    Returns true if a default model is set for the given meter.
    """
    client = get_mongo_client()
    query = {"refMeter": meter_id, "isDefault": True}
    model_meta = client[DB_NAME][MODEL_COLCTN].find_one(query)
    return model_meta is not None


def set_default_model(meter_id: str, algorithm: str):
    """
    Sets the isDefault property of the specified model to true
    and sets it to false for all other models of the same meter.
    """
    client = get_mongo_client()
    query = {"refMeter": meter_id}
    update = {"$set": {"isDefault": False}}
    client[DB_NAME][MODEL_COLCTN].update_many(query, update)

    query = {"refMeter": meter_id, "algorithm": {"$regex": algorithm, "$options": "i"}}
    update = {"$set": {"isDefault": True}}
    client[DB_NAME][MODEL_COLCTN].update_one(query, update)


def update_or_create_model_meta(properties, set_default=False):
    """
    Updates or creates a model meta document in the database.
    :param properties: The properties of the model meta document.
    :param setDefault: If true, this model is set as default for forecasting.
    """
    client = get_mongo_client()
    exists = client[DB_NAME][MODEL_COLCTN].find_one({"_id": properties["_id"]})
    if not exists:
        properties["dateCreated"] = properties["dateModified"]
        client[DB_NAME][MODEL_COLCTN].insert_one(properties)
    else:
        if set_default is not None:
            properties["isDefault"] = True
        client[DB_NAME][MODEL_COLCTN].update_one(
            {"_id": properties["_id"]}, {"$set": properties}
        )

    if set_default or (
        not has_default(properties["refMeter"])
        and properties["algorithm"] == DEFAULT_ALGORITHM
    ):
        set_default_model(properties["refMeter"], properties["algorithm"])


def get_models(meter_id):
    """
    Returns all models for the given meter.
    """
    client = get_mongo_client()
    query = {"refMeter": meter_id} if meter_id else {}
    models = list(client[DB_NAME][MODEL_COLCTN].find(query))
    return models


def get_model_meta(meter_id: str, algorithm: str):
    """
    Returns the model meta document for the given meter and algorithm.
    """
    client = get_mongo_client()
    query = {"refMeter": meter_id, "algorithm": {"$regex": algorithm, "$options": "i"}}
    model_meta = client[DB_NAME][MODEL_COLCTN].find_one(query)
    if model_meta is None:
        raise ResourceNotFoundException(
            "No model found for the given meter and algorithm."
        )
    return model_meta


def delete_model_meta(meter_id: str, algorithm: str, model_id: str = None):
    """
    Deletes the model meta document for the given meter and algorithm if it exists.
    Uses the model_id instead if it is specified.
    Returns the deleted document.
    """
    client = get_mongo_client()
    if model_id is not None:
        query = {"_id": model_id}
    else:
        query = {
            "refMeter": meter_id,
            "algorithm": {"$regex": algorithm, "$options": "i"},
        }
    model_meta = client[DB_NAME][MODEL_COLCTN].find_one(query)
    client[DB_NAME][MODEL_COLCTN].delete_one(query)
    return model_meta


def load_model_binary(meter_id: str, algorithm: str):
    """
    Loads the model binary from disk using the file path stored in the database.
    :param meter_id: The meter id.
    :param algorithm: The algorithm used for training the model.
    """
    model_meta = get_model_meta(meter_id, algorithm)
    if not os.path.isfile(model_meta["fpath"]):
        raise DataProcessingException("Could not find model on disk.")
    with open(model_meta["fpath"], "rb") as f:
        model = joblib.load(f)

        # Model weights are stored in a separate file for torch models, as
        # joblib seems to not automatically save torch models.
        if hasattr(model, "uses_torch") and model.uses_torch:
            # model.model.model accesses sequentially:
            # This project's model wrapper -> Dart's model wrapper -> the actual model
            weights_path = model_meta["fpath"].replace(".pkl", ".pth")
            model.model.model = model.model._create_model(model.model.train_sample)
            model.model.model.load_state_dict(torch_load(weights_path))

        return model


def delete_model_binary(
    model_meta: str = None, meter_id: str = None, algorithm: str = None
):
    """
    Deletes the model binary from disk using the file path in model_meta.
    If model_meta is not provided, it will be queried from the database using
    the specified meter_id and algorithm.
    :param model_meta: The model entry from the database.
                       If none, it will be queried from the database.
    :param meter_id: The meter id.
    :param algorithm: The algorithm used for training the model.
    """
    if model_meta is None:
        model_meta = get_model_meta(meter_id, algorithm)
    if os.path.isfile(model_meta["fpath"]):
        try:
            os.remove(model_meta["fpath"])
        except:
            raise DeleteResourceException("Could not delete model binary.")


def save_model(model, results: dict, set_default: bool = False, comment: str = ""):
    """
    Creates an entry for the model in the database and writes the model
    binary to disk.
    :param model: The model binary to be saved.
    :param results: The evaluation results. Should contain the fields
                    "metrics", "predicted", "actual"
    :param set_default: If true, this model is set as default for forecasting.
    :param comment: A comment for the model.
    """
    algorithm = model.name
    meter_id = model.meter_id
    model_id = get_model_id(meter_id, algorithm)
    date_str = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    model_path = get_model_path(meter_id, algorithm)

    try:
        with open(model_path, "wb") as fh:
            joblib.dump(model, fh)

        # Joblib does not automatically save torch model weights
        if hasattr(model, "uses_torch") and model.uses_torch:
            weights_path = model_path.replace(".pkl", ".pth")
            torch_save(model.model.model.state_dict(), weights_path)

        input_attributes = ["waterConsumption"]
        if model.has_covariates:
            input_attributes += list(results["testCovariates"].keys())

        model_meta = {
            # Standard MLModel fields
            "_id": model_id,
            "id": model_id,
            "algorithm": algorithm,
            "dateModified": date_str,
            "mlFramework": get_framework(algorithm),
            "description": f"{algorithm} model for meter {meter_id} to create a 24 hour forecast.",
            # Non-standard fields
            "refMeter": meter_id,
            "evaluation": results,
            "fpath": model_path,
            "isModelValid": True,
            "hyperparameters": model.hyperparameters,
            "inputAttributes": input_attributes,
            "comment": comment,
        }
        update_or_create_model_meta(model_meta, set_default=set_default)
        return model_id
    except Exception as e:
        print(e)

        # Clean up
        if os.path.isfile(model_path):
            os.remove(model_path)
        raise SaveResourceException("Error when trying to save model")


def is_orphan(meter_id):
    """
    Checks if the given meter_id represents or contains orphan virtual meters.
    An orphan vm is a virtual meter that has no submeters.
    """
    if is_virtual(meter_id):
        vm = get_virtual_meter(meter_id)
        if len(vm["submeterIds"]) == 0:
            return True
    return False


def get_orphans(meter_id):
    """
    Returns a list of all orphan virtual meters that are contained in the given meter.
    """
    orphans = []
    if is_virtual(meter_id):
        meter = get_virtual_meter(meter_id)
        orphans = [m for m in meter["submeterIds"] if is_orphan(m)]
    return orphans


def raise_if_orphan(meter_id):
    if is_orphan(meter_id):
        raise MethodNotAllowedException(
            "Cannot train model for orphan virtual meter, as it has no submeters. "
            + " Please delete this virtual meter and create a valid one."
        )
    elif len(get_orphans(meter_id)) > 0:
        raise MethodNotAllowedException(
            "Cannot train model for virtual meter, as it contains orphan submeters "
            + "that do not refer to any submeters themselves. "
            + "Please delete these virtual meters and create valid ones."
        )


def get_physical_meter_ids_of(meter_id):
    """
    Returns a list of all physical meters that are submeters of the given virtual meter.
    This includes the provided meter_id if it represents a physical meter.
    """
    meters = []
    if is_virtual(meter_id):
        vm = get_virtual_meter(meter_id)
        for submeter in vm["submeterIds"]:
            meters.append(submeter)
            if is_virtual(submeter):
                meters.extend(get_physical_meter_ids_of(submeter))
    else:
        meters.append(meter_id)
    return meters


def raise_if_not_disjoint(meter_ids):
    """
    Raises an exception if the given virtual meters' submeters overlap at any level.
    An overlap indicates that the submeters are not disjoint.
    """
    submeters = []
    for meter_id in meter_ids:
        submeters.extend(get_physical_meter_ids_of(meter_id))
    if len(submeters) != len(set(submeters)):
        counts = pd.Series(submeters).value_counts()
        duplicates = counts[counts > 1].index.tolist()
        raise BadRequestException(
            "Cannot create virtual meter from a set of submeters that contain duplicates."
            + "Please specify a disjoint set of submeters, the following submeters are duplicates: "
            + str(duplicates)
        )


#### Methods that do not directly access the database ####
def is_virtual(meter_id: str):
    """
    Returns True if the given meter id represents a virtual meter.
    """
    virtual_pattern = re.compile(r"^urn:ngsi-ld:(virtualMeter):[a-zA-Z0-9\.\-_+:]+$")
    physical_pattern = re.compile(r"^urn:ngsi-ld:(Device):[a-zA-Z0-9\.\-_+:]+$")
    if virtual_pattern.match(meter_id):
        return True
    elif physical_pattern.match(meter_id):
        return False
    else:
        raise DataProcessingException(
            "Could not determine if meter is virtual or physical for meter id "
            + meter_id
        )


def parse_vm_details(data: dict):
    """
    Checks and extracts the details of a virtual meter from raw request data and adds a new id.
    :param data: dict of raw data that contains the details of the virtual meter, but possibly more too.
    """
    vm_id = generate_vm_id(name=data.get("name", None))
    data["id"] = vm_id
    vm_details = VirtualMeterSchema().load(data, unknown=EXCLUDE)
    vm_details["submeterIds"] = list(vm_details["submeterIds"])
    vm_details["supermeterIds"] = []
    return vm_details


def parse_train_args(data: dict, algorithm: str = None):
    """
    Extracts the training options from raw request data. If the algorithm is given separately,
    for example because it is part of the URL, it is added to the train_options separately.
    :param data: dict of raw request parameters that contains the training options.
                 properties that are not part of TrainArgumentsSchema are ignored.
    :param algorithm: The algorithm that should be used for training.
    """
    train_options = TrainArgumentsSchema().load(data, unknown=EXCLUDE)
    if algorithm:
        train_options["algorithm"] = algorithm
    return train_options


def get_model_id(meter_id: str, algorithm: str):
    return f"{meter_id}:MLModel:{algorithm}"


def get_model_path(meter_id: str, algorithm: str):
    """
    Returns the file path for a model binary based on the meter id and algorithm.
    """
    model_id = get_model_id(meter_id, algorithm)
    return os.path.join(MODEL_FOLDER, f"{model_id.replace(':','_')}.pkl")


def get_framework(algorithm):
    """Returns the framework used for the given algorithm."""
    algorithm = algorithm.strip().lower()
    if algorithm in [
        "prophet",
        "autoarima",
        "tripleexponentialsmoothing",
        "xgboost",
        "nbeats",
        "temporalfusiontransformer",
    ]:
        return "darts"
    else:
        raise ValueError(f'Cannot find framework of algorithm "{algorithm}"')


def update_train_time(algorithm: str, train_time: float):
    """
    Updates the statistics in the database for how long training takes.
    The estimate is an exponential moving average.
    :param train_time: Training time for a single run in seconds.
    """
    algorithm = algorithm.strip().lower()
    client = get_mongo_client()
    query = {"_id": f"trainingTimeExponentialMovingAverage_{algorithm}"}

    estimate_prev = client[DB_NAME][STATS_COLCTN].find_one(query)

    alpha = 0.15
    if estimate_prev:
        estimate_new = estimate_prev["value"] * (1 - alpha) + train_time * alpha
    else:
        estimate_new = train_time

    update = {"$set": {"value": estimate_new, "algorithm": algorithm}}
    client[DB_NAME][STATS_COLCTN].update_one(query, update, upsert=True)


def get_train_time(algorithm: str):
    """
    Returns the estimated training time for the given algorithm.
    The estimate is an exponential moving average of the past training times
    plus a constant overhead of 1 minute.
    """
    client = get_mongo_client()
    query = {"_id": f"trainingTimeExponentialMovingAverage_{algorithm}"}
    estimate = client[DB_NAME][STATS_COLCTN].find_one(query)

    if estimate:
        return ceil(estimate["value"] / 60) + 1
    else:
        return None


def select_reference_days(forecast_date, meter_id):
    """
    Selects water consumption time series that serve as a historical reference
    for the given forecast date, e.g. for plausibility checking.
    :param forecast_date: The date for which the forecast is made and for
                          which reference days and values should be found.
    :param meter_id: The id of the meter for which the forecast is made.
    """
    df_measurements = get_measurements(meter_id, aggregate=True, diff=False)
    reference_deltas = {
        f"prevDay": timedelta(days=1),
        f"prevWeek": timedelta(days=7),
        f"prevMonth": timedelta(days=4 * 7),
    }
    reference_data = {}
    for name, delta in reference_deltas.items():
        ref_date = pd.Timestamp(get_start_of_day(forecast_date)) - delta
        ref_values = get_values_of_ref_day(df_measurements, ref_date)
        if ref_values is not None:
            reference_data[name] = ref_values
    return reference_data


def get_values_of_ref_day(df_measurements, date):
    """Selects the water consumption values of a reference day for the hours 1-24."""
    midnight = pd.Timestamp(get_start_of_day(date))
    start = midnight + pd.Timedelta(hours=1)
    end = start + timedelta(hours=23)
    values = df_measurements[start:end].values.flatten().tolist()
    if len(values) == 24:
        return values
