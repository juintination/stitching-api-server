# Flask Image Stitching API

## Description
This Flask application provides an API for extracting frames from videos, enhancing them using SRCNN (Super-Resolution Convolutional Neural Network), and stitching them into a panoramic image.

## Features
- Video frame extraction
- Image enhancement using SRCNN (3-layer and 4-layer models)
- Panoramic image stitching
- CORS support
- GPU acceleration (if available)

## Requirements
- Python 3.7+
- Flask
- Flask-RESTful
- Flask-CORS
- OpenCV (cv2)
- PyTorch
- NumPy
- Werkzeug

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required packages:
   ```
   pip install flask flask-restful flask-cors opencv-python-headless torch numpy werkzeug
   ```

3. Ensure you have the SRCNN model files:
   - `./model/srcnn_x2.pth.tar`
   - `./model/srcnn_x2_4.tar`

## Usage

1. Start the server:
   ```
   python app.py
   ```
   The server will start on `http://0.0.0.0:5050` by default.

2. API Endpoints:
   - `GET /`: Root endpoint to check if the API is working
   - `POST /convert`: Upload a video for frame extraction, enhancement, and stitching

3. To use the `/convert` endpoint:
   - Send a POST request with a video file in the body
   - The API will return a stitched panoramic image

## Configuration
- `UPLOAD_FOLDER`: Directory for storing processed images
- `PROCESSING_FOLDER`: Temporary directory for processing
- `TARGET_SIZE`: Target size for image resizing
- `PORT`: Server port (default: 5050)
- `MODEL_PATH`: Path to the 3-layer SRCNN model
- `MODEL_4LAYER_PATH`: Path to the 4-layer SRCNN model

## GPU Acceleration
The application automatically detects if a GPU is available and uses it for processing. Check the console output to see if GPU acceleration is enabled.

## Error Handling
The application includes comprehensive error handling and logging. Check the console output for detailed error messages and logs.

## CORS Configuration
CORS is configured to allow requests from the origin specified in the `ALLOWED_ORIGIN` environment variable. If not set, it allows all origins (`*`).

## Contributing
Contributions to this project are welcome. Please ensure to update tests as appropriate.

## License
This project is licensed under the Apache License 2.0. See the LICENSE file for details.

## Author
Bochan Kang (WellshCorgi)

## Contact
For any queries, please contact the Hongik University Software Department.
