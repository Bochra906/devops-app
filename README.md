# devops-app

# Description
The car damage detection application detects damages on used cars.
It contains two microservices: one for training the mask r-cnn model, and the other one for performing the inference process.
After detecting the damages and adding the masks, the result images are stored in Azure Storage Blob.
Our application is developed using python and FastAPI.

# Workflow
![workflow](https://user-images.githubusercontent.com/60546216/212982436-fe8746cf-be5b-4a8b-8596-23635e59d6dc.png)

# Docker
We dockerized the training and inference microservices and pushed the images to Dockerhub.

![dockerhub](https://user-images.githubusercontent.com/60546216/213249808-ead8797c-be29-4003-86ee-00187b6924ca.png)
