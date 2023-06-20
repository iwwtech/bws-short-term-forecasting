#!/bin/bash
# Note: If this script is not automatically executed, make sure that the line endings are linux style
echo "Populating MongoDB with data..."

# Load meta data
mongoimport --username $MONGO_INITDB_ROOT_USERNAME --password $MONGO_INITDB_ROOT_PASSWORD --authenticationDatabase admin \
--db $MONGO_INITDB_DATABASE --collection devices --file /docker-entrypoint-initdb.d/data/example_pm_meta.json --jsonArray

# Load measurements
mongoimport --username $MONGO_INITDB_ROOT_USERNAME --password $MONGO_INITDB_ROOT_PASSWORD --authenticationDatabase admin \
--db $MONGO_INITDB_DATABASE --collection deviceMeasurements --file /docker-entrypoint-initdb.d/data/example_pm_measurements.json --jsonArray

echo "Populating MongoDB done."
