# Image Stitching API (Django, Flask, FastAPI Versions)

## Project Overview
This project provides an API that extracts frames from a video, enhances the images using SRCNN (Super-Resolution Convolutional Neural Network), and stitches them into a panoramic image. It is implemented in three versions: Django, Flask, and FastAPI.

## 주요 기능
- Video frame extraction
- Image enhancement using SRCNN (3-layer and 4-layer models)
- Panoramic image stitching
- GPU acceleration support (if available)

## Requirements
- Python 3.7+
- OpenCV (cv2)
- PyTorch
- NumPy
- Django / Flask / FastAPI (depending on the chosen version)

## Installation and Setup
1. Clone the repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required packages:
   ```
   pip install django flask fastapi opencv-python-headless torch numpy
   ```

3. repare the SRCNN model files:
   - 3-layer model: `MODEL_PATH`
   - 4-layer model: `MODEL_4LAYER_PATH`

## 사용 방법
### Django
1. Start the server: `python manage.py runserver`
2. API Endpoints:
   - GET `/`: Check API status
   - POST `/convert/`: Upload and process video

### Flask
1. Start the server: `python app.py`
2. API Endpoints:
   - GET `/`: Check API status
   - POST `/convert/`: Upload and process video

### FastAPI
1. Start the server: `uvicorn main:app --host 0.0.0.0 --port 5050`
2. API Endpoints:
   - GET `/`: Check API status
   - POST `/convert/`: Upload and process video

## License
Apache License 2.0

