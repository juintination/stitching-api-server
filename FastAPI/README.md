# Image Stitching API

## Description
This FastAPI application provides an API for extracting frames from videos, enhancing them using SRCNN (Super-Resolution Convolutional Neural Network), and stitching them into a panoramic image. It also includes functionality for users to upload images for training data.

## Features
- Video frame extraction
- Image enhancement using SRCNN
- Panoramic image stitching
- User image upload for training data
- CORS support

## Requirements
- Python 3.7+
- FastAPI
- OpenCV (cv2)
- PyTorch
- NumPy
- uvicorn

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required packages:
   ```
   pip install fastapi opencv-python-headless torch numpy uvicorn python-multipart
   ```

3. Ensure you have the SRCNN model files:
   - `./model/srcnn_x2.pth.tar`
   - `./model/srcnn_x2_4.tar`

## Usage

1. Start the server:
   ```
   # Use OpenCV lib
   python main.py

   # Use Stitching lib
   python st_main.py
   ```
   The server will start on `http://0.0.0.0:5050`.

2. API Endpoints:
   - `GET /`: Root endpoint to check if the API is working
   - `POST /convert`: Upload a video for frame extraction, enhancement, and stitching

3. To use the `/convert` endpoint:
   - Send a POST request with a video file in the body
   - The API will return a stitched panoramic image

4. To use the `/upload` endpoint:
   - Send a POST request with an image file in the body
   - The API will save the image for future training purposes

## Configuration
- `UPLOAD_FOLDER`: Directory for storing processed images
- `UPLOAD_USER_DONATE`: Directory for storing user-uploaded images
- `PROCESSING_FOLDER`: Temporary directory for processing
- `TARGET_SIZE`: Target size for image resizing
- `MODEL_PATH`: Path to the 3-layer SRCNN model
- `MODEL_4LAYER_PATH`: Path to the 4-layer SRCNN model

## Error Handling
The application includes comprehensive error handling and logging. Check the console output for detailed error messages and logs.

## CORS Configuration
CORS is configured to allow requests from the origin specified in the `ALLOWED_ORIGIN` environment variable. If not set, it allows all origins (`*`).

## Contributing
Contributions to this project are welcome. Please ensure to update tests as appropriate.

## License
This project is licensed under the Apache License 2.0. See the LICENSE file for details.

## Contact
For any queries, please contact the Hongik University Software Department Gift-Set team.
