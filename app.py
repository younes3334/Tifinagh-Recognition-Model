# we will now deploy our model using FastAPI and uvicorn
# we will use the same model under the name "model_tifinagh.h5" for the Character Recognition

# import the necessary packages
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import uvicorn
import numpy as np
import cv2
import tensorflow as tf
from keras.models import load_model
import keras
import os
# initialize our FastAPI app
app = FastAPI()

# load the model
model = load_model("model_tifinagh.h5")
labels = []
dataset_dir = r"dataset"
# the firsl folder is "AMHCD_64", it contains folders of images with labels
# let's read the images and labels (the labels are the names of the folders)
images_dir = os.path.join(dataset_dir, "AMHCD_64")

for folder in os.listdir(images_dir):
    labels.append(folder)

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

# fit the label encoder to the labels
label_encoder.fit_transform(labels)

main_labels = []
with open(r"C:\Users\Asus\Desktop\Ultimate Projects\Tfinagh Recognetion model\Tifinagh-Recognition-Model\dataset\labels\33-common-latin-tifinagh.txt", "r") as f:
    for line in f:
        main_labels.append(line.split()[0])

# now we will take the corresponding labels from the file under :"C:\Users\Asus\Desktop\Ultimate Projects\Tfinagh Recognetion model\Tifinagh-Recognition-Model\dataset\labels\sorted-33-common-tifinagh.txt"
corresponding_labels = []
with open(r"C:\Users\Asus\Desktop\Ultimate Projects\Tfinagh Recognetion model\Tifinagh-Recognition-Model\dataset\labels\sorted-33-common-tifinagh.txt", "r", encoding="utf8") as f:
    for line in f:
        corresponding_labels.append(line.split()[0])
labels_dict = dict(zip(main_labels, corresponding_labels))



# define a predict function as an endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):

    # read the image file using cv2
    image = await file.read()
    image = np.frombuffer(image, np.uint8)

    image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)

    # move the gaussian noises
    image = cv2.GaussianBlur(image, (5, 5), 0)

    # resize the images to 64x64
    image = cv2.resize(image, (64, 64)) 

    # convert the images to numpy arrays
    image = np.array(image)


    # reshape the image to be 4D
    image = image.reshape(-1, 64, 64, 1)

    # predict the image
    y_pred = model.predict(image)
    y_pred = np.argmax(y_pred, axis=1)


    # return the prediction with the probability of the prediction in precentage :
    return {"prediction": labels_dict[labels[y_pred[0]]], "probability": str(model.predict(image)[0][y_pred[0]] * 100) + "%"}


@app.get("/")
def home():
    return {"health_check": "OK", "model_version": "__0.0.1__"}

# run the app using uvicorn with the following command:
# uvicorn app:app --reload



