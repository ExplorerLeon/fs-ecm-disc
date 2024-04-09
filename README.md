# fs-ecm-disc
discovery and testing of ecm project

Uses the finalize PubSub notification from Cloud Storage

gcloud functions deploy python-create-frp-function \
--gen2 \
--runtime=python312 \
--region=europe-west2 \
--source=. \
--entry-point=main \
--trigger-event-filters="type=google.cloud.storage.object.v1.finalized" \
--trigger-event-filters="bucket=ecm_automation_disc"


Cloud Build API
Artifact Registry API
Eventarc API
Cloud Run Admin API
Cloud Logging API
Cloud Pub/Sub API
Cloud Functions API
Cloud Storage API


Service Account:
PROJECT_ID=$(gcloud config get-value project) --> assetinsure-surety-data-models
PROJECT_NUMBER=$(gcloud projects list --filter="project_id:$PROJECT_ID" --format='value(project_number)') --> 489404383902

SERVICE_ACCOUNT=$(gsutil kms serviceaccount -p $PROJECT_NUMBER)

Same result to get Cloud Storage Service Agent `gcloud storage service-agent --project=assetinsure-surety-data-models`

gcloud projects add-iam-policy-binding assetinsure-surety-data-models \
  --member serviceAccount:service-489404383902@gs-project-accounts.iam.gserviceaccount.com \
  --role roles/pubsub.publisher

I think this one:
gcloud projects add-iam-policy-binding assetinsure-surety-data-models \
  --member serviceAccount:489404383902-compute@developer.gserviceaccount.com \
  --role roles/pubsub.publisher


#TODO: need to investigate the service account and service agent linked to pub sub cloud storage and cloud functions and permissions



HTTP Cloud Function
gcloud functions deploy python-http-create-frp-function \
--gen2 \
--runtime=python312 \
--region=europe-west2 \
--source=. \
--entry-point=main \
--trigger-http \
--allow-unauthenticated

--no-allow-unauthenticated
Then run like this:
curl  -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
  https://europe-west2-assetinsure-surety-data-models.cloudfunctions.net/python-private-http-create-frp-function
