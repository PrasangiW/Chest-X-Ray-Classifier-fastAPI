# CHEST X-RAY CLASSIFIER

The Chest X-Ray Classifier API is a FastAPI application designed to classify chest X-ray images into three categories: COVID-19, Viral Pneumonia, and Normal. Leveraging a TensorFlow Lite model, this API allows users to upload X-ray images and receive real-time predictions about the image's classification. It provides a simple and efficient way to integrate X-ray image analysis into applications through easy-to-use HTTP endpoints.

Deploy Fastapi project using render :https://chest-x-ray-classifier-fastapi.onrender.com

![Screenshot 2024-09-14 102830](https://github.com/user-attachments/assets/7479d679-9a9e-4ebb-836c-bfa188561a5e)

# Installation

1. git clone https://github.com/PrasangiW/Chest-X-Ray-Classifier-fastAPI.git

2. cd Chest X-Ray Classifier

3. Create and Activate a Virtual Environment (optional but recommended):
    python -m venv venv
    venv\Scripts\activate

4. Install Dependencies:
    pip install -r requirements.txt

5. Start the FastAPI Server:
    uvicorn main:app --reload   *This will start the server on http://127.0.0.1:8000.*


#Accessing the API

Interactive API Documentation: Open your browser and navigate to http://127.0.0.1:8000/docs to interact with the API. This page provides an interactive interface for testing the API endpoints.


# Endpoints

 Root Endpoint (GET /)
 Description: Returns a welcome message.
 Request:
curl -X 'GET' \
  'http://127.0.0.1:8000/' \
  -H 'accept: application/json'
 Response:
{
  "message": "Welcome to the Chest X-Ray Classifier API"
}

![Screenshot 2024-09-14 103113](https://github.com/user-attachments/assets/de81ec30-533b-4be1-842d-eb304330c4e6)

# Status Endpoint (GET /status)

 Description: Provides the status of the service.
 Request:
curl -X 'GET' \
  'http://127.0.0.1:8000/status' \
  -H 'accept: application/json'
 Response:
{
  "message": "Service is running"
}
![Screenshot 2024-09-14 103439](https://github.com/user-attachments/assets/4dd1d821-6535-4deb-b73a-945429c00bcc)


# Predict Endpoint (POST /predict)
 Description: Upload a chest X-ray image to get a prediction.
 Request:
ccurl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@76052f7902246ff862f52f5d3cd9cd_big_gallery.jpg;type=image/jpeg'
 Response:
{
  "class_name": "Normal",
  "probability": 0.523370623588562
}

# Features

- **Image Classification**: Classify X-ray images into predefined categories.
- **API Endpoints**: Interact with the service via simple HTTP endpoints.
- **deploy**:fastapi deploy in to render and anyone can access and use the model
