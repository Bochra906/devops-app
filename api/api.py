from fastapi import FastAPI , Request, HTTPException
from pydantic import BaseModel
import boto3
from io import BytesIO
import numpy as np 
from typing import Union
import tensorflow as tf
from importlib import reload
import car_damage_detection as cd
import os
import cv2
from PIL import Image
import requests
import pymongo
from pymongo import MongoClient
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient


app = FastAPI()


class ImageIn(BaseModel):
    image_link: str
    user_id: Union[str, None] = None
    car_id: Union[str, None] = None

class ImageOut(BaseModel):
    image_id: str
    user_id: Union[str, None] = None
    car_id: Union[str, None] = None

class ImageMongo(BaseModel):
    image_blob_link: str
    nb_damages: int
    
async def create_image(image_blob_link, nb_damages):
    new_image= ImageMongo(image_blob_link= image_blob_link, nb_damages=nb_damages)
    return dict(new_image)

async def write_image_to_blob(image , key ):
    try: 
        account_url = "https://bochrasinda.blob.core.windows.net"
        default_credential = DefaultAzureCredential()
        # Create the BlobServiceClient object
        blob_service_client = BlobServiceClient(account_url, credential=default_credential)
    except Exception as ex:
        print('Exception:')
        print(ex)
        
    #save image to local path
    image.save(f"./results/{key}", format="PNG")

    # Create a blob client using the local file name as the name for the blob
    blob_client = blob_service_client.get_blob_client(container="images", blob=f"{key}.png")



    # Upload the created file
    with open(file=f"./results/{key}", mode="rb") as data:
        blob_client.upload_blob(data)
    
    image_url = f"https://bochrasinda.blob.core.windows.net/images/{key}.png"
    return image_url


#-----------------------------------------------------
# prepare the model and load the weights for inference
#-----------------------------------------------------
reload(cd.visualize)
# Create model in inference mode
with tf.device(cd.DEVICE):
    model = cd.modellib.MaskRCNN(mode="inference", model_dir=cd.MODEL_DIR, config=cd.config)

# load the last trained model
weights_path = "mask_rcnn_scratch_0004.h5"
#weights_path = model.find_last()

# Load weights
model.load_weights(weights_path, by_name=True)

# layer types to display
LAYER_TYPES = ['Conv2D', 'Dense', 'Conv2DTranspose']

# Get layers
layers = model.get_trainable_layers()
layers = list(filter(lambda l: l.__class__.__name__ in LAYER_TYPES, 
                layers))

# Load Validation Dataset
dataset = cd.custom.CustomDataset()
dataset.load_custom(cd.dataset_DIR,'val')
dataset.prepare()

async def inference (image_link):
    global model

    # read the image from the url
    response = requests.get(image_link)
    img = Image.open(BytesIO(response.content))
    image = np.array(img, dtype="uint8")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run object detection
    results = model.detect([image], verbose=1)

    # Display results
    ax = cd.get_ax(1)
    r = results[0]
    result , nb_damages = cd.visualize.display_instances(
        image, r['rois'], r['masks'], r['class_ids'],
        dataset.class_names, r['scores'], ax=ax,
        title="Prediction"
    )
    return result , nb_damages

@app.post("/detectCarDamage" , response_model=ImageOut, tags=["damage"])
async def car_damage_detection(request:Request):
    
    data = await request.json()

    if len(data.get("image_link")) == 0 :
        raise HTTPException(status_code=404, detail="Empty link")

    image = ImageIn(image_link=data.get("image_link"))

    #apply inference code
    result , nb_damages = await inference(image.image_link)

    if result is None:
        raise HTTPException(status_code=404, detail="Invalid inference result")
    
    key = image.image_link.split('/')[-1]
        
    
    image_url = write_image_to_blob( result, key )

    response = requests.get(image_url)
    if response.status_code != 200:
        raise HTTPException(status_code=404, detail="Invalid URL")

    #connection with mongodb
    try:
        client = MongoClient("mongodb+srv://devops-project:devopsproject@cluster0.lp8tgkx.mongodb.net/test", 80)
    except ConnectionError:
        raise pymongo.errors.ConnectionFailure

    db = client.local
    collection = db.car_damage_images
    new_image = await create_image(image_url , nb_damages)
    x = collection.insert_one(new_image)
    output = ImageOut(image_id=str(x.inserted_id), user_id=data.get("user_id") , car_id=data.get("car_id") )
    return output