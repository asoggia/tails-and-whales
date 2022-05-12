from fastapi import FastAPI
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
import cv2
import matplotlib.image as mpimg
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return {"greeting": "Hello Whales !"}

@app.get("/predict/{name_img}")
async def predict(name_img):
    img = mpimg.imread("image/" + name_img)  # changer le path !!!
    img = cv2.resize(img, dsize=(64, 64), interpolation= cv2.INTER_LINEAR)

    X = np.array(img)
    X = preprocess_input(X)
    X = np.expand_dims(X, axis=0)

    reconstructed_model = load_model('model_7/') #attention changer le path
    prediction = reconstructed_model.predict(X)

    prediction = [float(i) for i in prediction[0]]
    reponse = {"prediction": prediction}
    return reponse
