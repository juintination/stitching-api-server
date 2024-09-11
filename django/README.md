# Django Image Stitching API

## Description
This Django application provides an API for extracting frames from videos, enhancing them using SRCNN (Super-Resolution Convolutional Neural Network), and stitching them into a panoramic image.

## Features
- Video frame extraction
- Image enhancement using SRCNN (3-layer and 4-layer models)
- Panoramic image stitching
- GPU acceleration (if available)

## Requirements
- Python 3.7+
- Django
- OpenCV (cv2)
- PyTorch
- NumPy

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required packages:
   ```
   pip install django opencv-python-headless torch numpy
   ```

3. Ensure you have the SRCNN model files in the correct location (as specified in your Django settings):
   - `MODEL_PATH` for the 3-layer SRCNN model
   - `MODEL_4LAYER_PATH` for the 4-layer SRCNN model

## Project Structure
- `views.py`: Contains the main logic for the API endpoints
- `urls.py`: Defines the URL patterns for the API
- `models.py`: Defines the SRCNN and SRCNN_4Layer models (not shown in the provided code)

## Configuration
Make sure to set the following in your Django settings:
- `UPLOAD_FOLDER`: Directory for storing processed images
- `PROCESSING_FOLDER`: Temporary directory for processing
- `MODEL_PATH`: Path to the 3-layer SRCNN model
- `MODEL_4LAYER_PATH`: Path to the 4-layer SRCNN model

## Usage

1. Start the Django development server:
   ```
   python manage.py runserver
   ```

2. API Endpoints:
   - `GET /`: Root endpoint to check if the API is working
   - `POST /convert/`: Upload a video for frame extraction, enhancement, and stitching

3. To use the `/convert/` endpoint:
   - Send a POST request with a video file in the body (key: 'file')
   - The API will return a stitched panoramic image

## GPU Acceleration
The application automatically detects if a GPU is available and uses it for processing. Check the console output to see if GPU acceleration is enabled.

## Error Handling
The application includes comprehensive error handling and logging. Check the console output for detailed error messages and logs.

## CSRF Protection
The `convert` view is exempt from CSRF protection to allow for easier API usage. Ensure proper security measures are in place when deploying to production.

## Contributing
Contributions to this project are welcome. Please ensure to update tests as appropriate.

## License
This project is licensed under the Apache License 2.0. See the LICENSE file for details.

## Contact
For any queries, please contact the Hongik University Software Department.

## Notes
- This README assumes that the Django project and app are already set up. You may need to adjust the instructions based on your specific project structure.
- Make sure to configure your Django settings (`settings.py`) appropriately, including database settings, installed apps, and middleware.
- For production deployment, follow Django's deployment best practices and consider using a production-grade server like Gunicorn with Nginx.
