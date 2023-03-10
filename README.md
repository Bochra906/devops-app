# devops-app

# Description
The car damage detection application detects damages on used cars.
It contains two microservices: one for training the mask r-cnn model, and the other one for performing the inference process.
After detecting the damages and adding the masks, the result images are stored in Azure Storage Blob.
Our application is developed using python and FastAPI.

This is an example of an image before and after inference process:
![before](https://user-images.githubusercontent.com/60546216/213252530-29088a8e-73cb-4577-ad09-4f1ec8f415c1.png)
![after](https://user-images.githubusercontent.com/60546216/213252546-acc25674-3fbe-497b-bfed-e5c9dba19714.png)

# Workflow
![workflow](https://user-images.githubusercontent.com/60546216/213250047-f7b16c3e-bdbf-4a1c-ab2d-dcd88f560ba3.png)

# Docker
We dockerized the training and inference microservices and pushed the images to Dockerhub.

![dockerhub](https://user-images.githubusercontent.com/60546216/213249808-ead8797c-be29-4003-86ee-00187b6924ca.png)
