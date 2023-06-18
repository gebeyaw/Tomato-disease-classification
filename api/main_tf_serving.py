from fastapi import FastAPI, File, UploadFile  
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import requests 

app = FastAPI()


 
#watch tf serving videos from codebasic

# to use the latest version of the saved_model
endpoint = "https://localhost:8501/v1/models/tomatoes_model:predict"

CLASS_NAMES = ["Tomato__Target_Spot","Tomato__Tomato_mosaic_virus","Tomato__Tomato_YellowLeaf__Curl_Virus",
               "Tomato_Bacterial_spot","Tomato_Early_blight","Tomato_healthy","Tomato_Late_blight",
               "Tomato_Leaf_Mold","Tomato_Septoria_leaf_spot","Tomato_Spider_mites_Two_spotted_spider_mite""a","b","c","d","e","f","g","h","i","j"]
@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data)-> np.array:
    image = np.array(Image.open(BytesIO(data)))
    img_batch = np.expand_dims(image,0)
    json_data = {
        "instance": img_batch.tolist() 
        }
    response = requests.post(endpoint,json= json_data)
    prediction = np.array(response.json()["predictions"][0])
    
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)
    return {
        'class': predicted_class,
        'confidence': float(confidence)
        
    }


@app.post("/predict")
async def predict(
    file:UploadFile = File(...)
):
    image = read_file_as_image(await file.read())





if __name__=="__main__":
    uvicorn.run(app,host='localhost',port =8000)
  
