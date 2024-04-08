#!/bin/bash

# Create a Google Cloud Storage bucket
gcloud storage buckets create gs://ecm_automation_p1_output_disc --location=europe-west2

# Copy the file to the bucket
gcloud storage cp ~/Downloads/input/FRP/Portfolio\ Monitoring\ \(Dec\ 2023\).xlsm gs://ecm_automation_disc/

gcloud storage cp ./input_data/FRPSep2023-FirstLoss-Change.xlsx gs://ecm_automation_disc/
