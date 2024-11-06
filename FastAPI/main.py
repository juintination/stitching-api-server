# filename: main.py
# author: Bochan Kang
# date: 2024-11-06
# version: 1.0
# description: Enhance FastAPI version

import os
import cv2
import uuid
import shutil
import numpy as np
import torch
from torch import nn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import logging
from fastapi.middleware.cors import CORSMiddleware

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Image Stitching API",
    description="비디오에서 프레임을 추출하고 초해상화 하여 이미지를 스티칭하는 API",
    version="1.0",
    contact={
        "name": "Hongik univ. Software Dept. Gift-Set team.",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=['ALLOWED_ORIGIN', '*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static configuration
UPLOAD_FOLDER = 'uploads'
UPLOAD_USER_DONATE = 'useruploads'
PROCESSING_FOLDER = 'processing'
TARGET_SIZE = 1000
MODEL_PATH = './model/srcnn_x2.pth.tar'
MODEL_4LAYER_PATH = './model/srcnn_x2_4.tar'

# GPU availability check and device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"[Server System] Using device: {device}")

# Directory creation function
def create_directory(directory):
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError as e:
        logger.error(f"[mkdir Error] Failed to create directory {directory}: {e}")
        raise

# Image resizing function
def resize_image(img, target_size=1000):
    try:
        h, w = img.shape[:2]
        if h > w:
            if h > target_size:
                w = int(w * (target_size / h))
                h = target_size
        else:
            if w > target_size:
                h = int(h * (target_size / w))
                w = target_size
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    except Exception as e:
        logger.error(f"[preprocess] Error during resize: {e}")
        return None

# Image stitching function
def stitch_images(images, output_path):
    try:
        cv2.ocl.setUseOpenCL(False)  # OpenCL 비활성화
        stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
        status, stitched_img = stitcher.stitch(images)
        if status == cv2.Stitcher_OK:
            logger.info("[Stitcher] Image stitching successful!")
            cv2.imwrite(output_path, stitched_img)
        else:
            logger.error(f"[Stitcher] Image stitching failed: {status}")
            handle_stitching_error(status)
    except cv2.error as e:
        error_msg = str(e)
        if "DLASCLS" in error_msg or "illegal value" in error_msg:
            logger.critical(f"[Stitcher] DLASCLS error occurred: {error_msg}")
            raise HTTPException(status_code=500, detail="Critical stitching error occurred")
        else:
            logger.error(f"[Stitcher] OpenCV error occurred: {e}")
    except Exception as e:
        logger.error(f"[Stitcher] Unexpected error occurred: {e}")

# Stitching error handling function
def handle_stitching_error(status):
    error_messages = {
        cv2.Stitcher_ERR_NEED_MORE_IMGS: "More images are needed.",
        cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "Homography estimation failed. Images are too different.",
        cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "Camera parameter adjustment failed.",
    }
    error_message = error_messages.get(status, "Unknown error occurred")
    logger.warning(f"[Stitcher error] {error_message}")
    raise HTTPException(status_code=400, detail=error_message)

# SRCNN model definition
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, (9, 9), (1, 1), (4, 4)),
            nn.ReLU(True)
        )
        self.map = nn.Sequential(
            nn.Conv2d(64, 32, (5, 5), (1, 1), (2, 2)),
            nn.ReLU(True)
        )
        self.reconstruction = nn.Conv2d(32, 1, (5, 5), (1, 1), (2, 2))
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = self.map(out)
        out = self.reconstruction(out)
        return out

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight.data, 0.0, (2 / (module.out_channels * module.weight.data[0][0].numel())) ** 0.5)
                nn.init.zeros_(module.bias.data)
        nn.init.normal_(self.reconstruction.weight.data, 0.0, 0.001)
        nn.init.zeros_(self.reconstruction.bias.data)

#4layer 기반 SRCNN 모델 정의
class SRCNN_4Layer(nn.Module):
    def __init__(self) -> None:
        super(SRCNN_4Layer, self).__init__()
        # Feature extraction layer.
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, (9, 9), (1, 1), (4, 4)),
            nn.ReLU(True)
        )

        # Non-linear mapping layer.
        self.map = nn.Sequential(
            nn.Conv2d(64, 32, (5, 5), (1, 1), (2, 2)),
            nn.ReLU(True),
            # 새로운 레이어 추가
            nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True)
        )

        # Rebuild the layer.
        self.reconstruction = nn.Conv2d(32, 1, (5, 5), (1, 1), (2, 2))
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = self.map(out)
        out = self.reconstruction(out)

        return out

    # The filter weight of each layer is a Gaussian distribution with zero mean and
    # standard deviation initialized by random extraction 0.001 (deviation is 0)
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight.data, 0.0, (2 / (module.out_channels * module.weight.data[0][0].numel())) ** 0.5)
                nn.init.zeros_(module.bias.data)
        nn.init.normal_(self.reconstruction.weight.data, 0.0, 0.001)
        nn.init.zeros_(self.reconstruction.bias.data)

# Load the model based on the chosen layers
def load_model(layer_choose):
    if layer_choose == 3:
        model = SRCNN()
        checkpoint = torch.load(MODEL_PATH, map_location=lambda storage, loc: storage)
    elif layer_choose == 4:
        model = SRCNN_4Layer()
        checkpoint = torch.load(MODEL_4LAYER_PATH, map_location=lambda storage, loc: storage)
    else:
        raise ValueError("Invalid model layer choice")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model.to(memory_format=torch.channels_last, device=device)

# Image enhancement function (using PyTorch SRCNN)
def enhance_image(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(image)
    y = y.astype(np.float32) / 255.0
    y_tensor = torch.from_numpy(y).unsqueeze(0).unsqueeze(0)
    y_tensor = y_tensor.to(device=device, memory_format=torch.channels_last, non_blocking=True)
    with torch.no_grad():
        y_enhanced_tensor = model(y_tensor).clamp_(0, 1.0)
    y_enhanced = y_enhanced_tensor[0, 0].cpu().numpy()
    y_enhanced = (y_enhanced * 255.0).clip(0, 255).astype(np.uint8)
    enhanced_image = cv2.merge([y_enhanced, cr, cb])
    enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_YCrCb2BGR)
    return enhanced_image

# Image loading and preprocessing function
def load_and_preprocess_image(filepath, target_size=1000):
    try:
        img = cv2.imread(filepath)
        if img is not None:
            img = resize_image(img, target_size)
            img = cv2.GaussianBlur(img, (5, 5), 0)
        return img
    except Exception as e:
        logger.error(f"[preprocess] Error during image load and preprocessing: {e}")
        return None

# Frame extraction and saving function
def extract_frames(video_path, output_dir, interval=1, enhance=False):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"[cv2 Error] Cannot open video file: {video_path}")
            raise HTTPException(status_code=400, detail="Cannot open video file")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * interval)
        create_directory(output_dir)
        
        model = None
        if enhance:
            model = load_model(layer_choose=4)

        frame_count = 0
        extracted_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                if enhance and model:
                    frame = enhance_image(frame, model)
                frame_filename = os.path.join(output_dir, f"frame_{extracted_count:04d}.jpg")
                cv2.imwrite(frame_filename, frame)
                extracted_count += 1
            frame_count += 1
        cap.release()
        
        logger.info(f"[OpenCV] {extracted_count} frames saved to {output_dir}") 
    except Exception as e:
        logger.error(f"[OpenCV] Error during frame extraction: {e}")
        raise HTTPException(status_code=500, detail="Error during frame extraction")

# Cleanup function to remove temporary files
def clean_up(directory):
    try:
        if os.path.exists(directory):
            shutil.rmtree(directory)
            logger.info(f"[Cleanup] Temporary directory {directory} removed.")
    except OSError as e:
        logger.error(f"[Cleanup Error] Failed to remove directory {directory}: {e}")


# API endpoints
@app.get("/", summary="root EndPoint", description="Check API is Working on Server.")
async def read_root():
    """
       루트 엔드포인트를 반환합니다.
    
    Returns:
        dict: 고정된 응답을 포함하는 딕셔너리
    """
    return {"name": "hongik", "type": "university"}

@app.post("/convert", summary="Convert Video", description="Processing uploaded Video to JPG pictures  and Return Stitching Image about pictures")
async def convert(file: UploadFile = File(...)):
    """
    업로드된 비디오 파일을 처리하여 스티칭된 이미지를 생성합니다.
    
    Args:
        file (UploadFile): 업로드된 비디오 파일

    Returns:
        FileResponse: 스티칭된 이미지 파일

    Raises:
        HTTPException: 처리 중 오류가 발생한 경우
    """
    # Log headers and file info
    logger.info(f"[Request] Headers: {file.headers}")
    logger.info(f"[Request] File name: {file.filename}, Content-Type: {file.content_type}, File size: {file.file.tell()}")

    unique_id = str(uuid.uuid4())
    processing_dir = os.path.join(PROCESSING_FOLDER, unique_id)
    frames_dir = os.path.join(processing_dir, 'frames')
    result_dir = os.path.join(UPLOAD_FOLDER, unique_id)

    try:
        create_directory(UPLOAD_FOLDER)
        create_directory(processing_dir)
        create_directory(frames_dir)
        create_directory(result_dir)

        video_path = os.path.join(processing_dir, file.filename)
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        extract_frames(video_path, frames_dir, interval=1, enhance=True)

        images = []
        for img_file in sorted(os.listdir(frames_dir)):
            img_path = os.path.join(frames_dir, img_file)
            img = load_and_preprocess_image(img_path, target_size=2000)
            if img is not None:
                images.append(img)

        result_path = os.path.join(result_dir, 'result_img.jpg')
        if len(images) > 0:
            logger.info('[Stitcher] Starting image stitching')
            stitch_images(images, result_path)
            response = FileResponse(result_path, media_type='image/jpeg', filename='result_img.jpg')
            clean_up(processing_dir)
            return response
        else:
            clean_up(processing_dir)
            raise HTTPException(status_code=400, detail="No images to stitch")

    except Exception as e:
        clean_up(processing_dir)
        logger.error(f"[Stitcher] Error during processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4050)
