*CHEST X-RAY CLASSIFIER*

The Chest X-Ray Classifier API is a FastAPI application designed to classify chest X-ray images into three categories: COVID-19, Viral Pneumonia, and Normal. Leveraging a TensorFlow Lite model, this API allows users to upload X-ray images and receive real-time predictions about the image's classification. It provides a simple and efficient way to integrate X-ray image analysis into applications through easy-to-use HTTP endpoints.

*Installation*

1. git clone 

2. cd Chest X-Ray Classifier

3. Create and Activate a Virtual Environment (optional but recommended):
    python -m venv venv
    venv\Scripts\activate

4. Install Dependencies:
    pip install -r requirements.txt

5. Start the FastAPI Server:
    uvicorn main:app --reload   *This will start the server on http://127.0.0.1:8000.*


*Accessing the API*

Interactive API Documentation: Open your browser and navigate to http://127.0.0.1:8000/docs to interact with the API. This page provides an interactive interface for testing the API endpoints.


*Endpoints*

# Root Endpoint (GET /)
# Description: Returns a welcome message.
# Request:
curl -X 'GET' 'http://127.0.0.1:8000/'
# Response:
{
  "message": "Welcome to the Chest X-Ray Classifier API"
}

*Status Endpoint (GET /status)*

# Description: Provides the status of the service.
# Request:
curl -X 'GET' 'http://127.0.0.1:8000/status'
# Response:
{
  "message": "Service is running"
}


*Predict Endpoint (POST /predict)*
# Description: Upload a chest X-ray image to get a prediction.
# Request:
curl -X 'POST' 'http://127.0.0.1:8000/predict' -F 'file=@path_to_your_image.jpg'
# Response:
{
  "class_name": "Covid"
}

*Features*

- **Image Classification**: Classify X-ray images into predefined categories.
- **API Endpoints**: Interact with the service via simple HTTP endpoints.
- **deploy**:fastapi deploy in to render and anyone can access and use the model