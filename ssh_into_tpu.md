
Authenticate GCP\

gcloud config set project <project-ID>

gcloud services enable tpu.googleapis.com
  
  
> export TPU_NAME=andrew-tpu \
> export ZONE=us-central1-a

> gcloud compute tpus tpu-vm ssh ${TPU\_NAME} --zone=${ZONE} 

> gcloud compute tpus tpu-vm describe ${TPU\_NAME}  --zone=${ZONE} 

1) Observe Host ID, and external IP, and post into config file\
2) Remove HostID from GCPKnownHosts in .ssh\
