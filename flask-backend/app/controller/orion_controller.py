import requests
from app.utils.constants import CORE_BASE_URL, ORION_BASE_URL
from app.utils.exception_handling import OrionCommunicationException

SERVICE_DESCR = "Service that offers hourly water demand forecast for the next 24 hours for specific locations."
HEADERS = {"Content-Type": "application/json"}


def register_service_with_orion():
    """
    Registers a service with the Orion Context Broker if it is not already registered.
    See also: 
    - https://github.com/telefonicaid/fiware-orion/blob/master/doc/manuals/orion-api.md#registration-payload-datamodel
    - https://swagger.lab.fiware.org/
    """
    ### Check if service is already registered
    registration_url = f"{ORION_BASE_URL}/registrations"
    response = requests.get(registration_url)
    if response.status_code == 200:
        registrations = response.json()
        for registration in registrations:
            if registration["description"] == SERVICE_DESCR:
                print(
                    "Service already registered with Orion. Registration ID:",
                    registration["id"],
                )
                return

    ### Register service
    payload = {
        "id": "short-term-water-demand-forecasting",
        "description": SERVICE_DESCR,
        "dataProvided": {
            # NOTE: that we refer to all meters as Devices, because
            # there is no official datatype corresponding to virtual meters.
            "entities": [{"idPattern": ".*", "type": "Device"}],
        },
        "provider": {"http": {"url": CORE_BASE_URL}},
    }

    response = requests.post(registration_url, headers=HEADERS, json=payload)
    if response.status_code == 201:
        registration_id = response.headers.get("Location").split("/")[-1]
        print(
            "Orion service registration successful. Registration ID:", registration_id,
        )
    else:
        print("Orion service registration failed. Status code:", response.status_code)


def replace_paranth(string):
    """
    Replaces parantheses in a string with underscores.
    Useful to convert e.g. "(m3)" to "_m3_", as Orion complains
    about parantheses in attributes.
    """
    return string.replace("(", "_").replace(")", "_")


def to_entity(results, meter_meta, model_meta):
    """
    Constructs an entity from results and meta information.
    Note that we use the simplified entity representation. See also: https://github.com/telefonicaid/fiware-orion/blob/master/doc/manuals/orion-api.md#simplified-entity-representation
    """

    entity = {
        "id": meter_meta["id"],
        "type": "Device",
        "description": meter_meta["description"],
        "mlModel": {
            "id": model_meta["id"],
            "type": "MLModel",
            "description": model_meta["description"],
            "dateModified": model_meta["dateModified"],
            "hyperparameters": model_meta["hyperparameters"],
            "inputAttributes": [
                replace_paranth(s) for s in model_meta["inputAttributes"]
            ],
            "evaluation": model_meta["evaluation"],
        },
        ## Note: results is a list of dicts where each dict contains the forecast for one hour
        "forecast": results,
    }

    if "address" in meter_meta:
        entity["address"] = meter_meta["address"]

    return entity


def create_or_update_entity(results, meter_meta, model_meta):
    """
    Creates or updates an entity with its context in Orion.
    See also: https://github.com/telefonicaid/fiware-orion/blob/master/doc/manuals/orion-api.md#create-entity-post-v2entities
    """
    entity = to_entity(results, meter_meta, model_meta)
    headers = {"Content-Type": "application/json"}
    url = f"{ORION_BASE_URL}/entities?options=upsert,keyValues"
    response = requests.post(url, headers=headers, json=entity)
    if response.status_code in range(200, 300):
        print(f"Entity creation for {entity['id']} successful.")
    else:
        msg = f"Entity creation for {entity['id']} failed. Status code: {response.status_code}"
        msg += f"\nResponse: {response.text}"
        print(msg)
        raise OrionCommunicationException(msg)


def delete_entity(meter_id):
    """Deletes an entity from Orion."""
    url = f"{ORION_BASE_URL}/entities/{meter_id}"

    response = requests.get(url)
    if response.status_code == 404:
        print(f"Orion entity {meter_id} does not exist thus was not deleted.")
        return

    response = requests.delete(url)
    if response.status_code == 204:
        print(f"Orion entity deletion for {meter_id} successful.")
    else:
        print(
            f"Orion entity deletion for {meter_id} failed. Status code:",
            response.status_code,
        )
