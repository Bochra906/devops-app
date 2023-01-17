# devops-app

# Description
The car damage detection application detects damages on used cars.
It contains two microservices: one for training the mask r-cnn model, and the other one for performing the inference process.
After detecting the damages and adding the masks, the result images are stored in Azure Storage Blob.
Our application is developed using python and FastAPI.

# Workflow

# Docker
We dockerized the training and inference microservices and pushed the images to Dockerhub.