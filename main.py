from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from PIL import Image
import io

# Load TFLite model
model_path = "model.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
class_names = ['Covid', 'Viral Pneumonia', 'Normal']

app = FastAPI()

class PredictionResponse(BaseModel):
    class_name: str

@app.get("/")
async def root():
    return {"message": "Welcome to the Chest X-Ray Classifier API"}

@app.get("/status")
async def get_status():
    return {"message": "Service is running"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    # Read image
    image = Image.open(io.BytesIO(await file.read()))

    # Preprocess image
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((input_shape[1], input_shape[2]))
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)

    # Make prediction
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])
    predicted_class_index = np.argmax(predictions, axis=1)
    predicted_class_name = class_names[predicted_class_index[0]]

    return PredictionResponse(class_name=predicted_class_name)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
