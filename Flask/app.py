# filename: app.py
# author: Bochan Kang (WellshCorgi)
# date: 2024-08-31
# version: 2.1
# description : Added a four-layer SRCNN model

import os
import cv2
import werkzeug
import uuid
import shutil
import numpy as np
import torch
from torch import nn
from flask import Flask, jsonify, send_file
from flask_restful import Resource, Api, reqparse
from flask_cors import CORS
from werkzeug.utils import secure_filename
import logging

# logging Config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

application = Flask(__name__)
api = Api(application)
CORS(application, resources={r"/*": {"origins": os.getenv('ALLOWED_ORIGIN', '*')}})
application.config['JSON_AS_ASCII'] = False

# Static - Config
UPLOAD_FOLDER = 'uploads'
PROCESSING_FOLDER = 'processing'
TARGET_SIZE = 1000
PORT = 5050
MODEL_PATH = './model/srcnn_x2.pth.tar'
MODEL_4LAYER_PATH = './model/srcnn_x2_4.tar'

# GPU 사용 가능 여부 확인 및 device 설정, GPU 사용 여부 표시
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == 'cuda':
    logger.info("[Server System] This Server Using GPU !!!")
else:
    logger.info("[Server System] Not using GPU. Using CPU.")

# 디렉토리 생성 함수
def create_directory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError as e:
        logger.error(f"[mkdir Error] 디렉토리 {directory} 생성 실패: {e}")
        raise

# 이미지 리사이즈 함수
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
        logger.error(f"[preprocess] Resize 중 오류 발생: {e}")
        return None

# 이미지 스티칭 함수
def stitch_images(images, output_path):
    try:
        stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
        status, stitched_img = stitcher.stitch(images)
        if status == cv2.Stitcher_OK:
            logger.info("[Stitcher] 이미지 스티칭 성공!")
            cv2.imwrite(output_path, stitched_img)
        else:
            logger.error(f"[Stitcher] 이미지 스티칭 실패: {status}")
            handle_stitching_error(status)
    except cv2.error as e:
        error_msg = str(e)
        if "DLASCLS" in error_msg or "illegal value" in error_msg:
            logger.critical(f"[Stitcher] DLASCLS 오류 발생: {error_msg}")
            raise SystemExit(1)
        else:
            logger.error(f"[Stitcher] OpenCV 오류 발생: {e}")
    except Exception as e:
        logger.error(f"[Stitcher] 예상치 못한 오류 발생: {e}")

# 스티칭 오류 처리 함수
def handle_stitching_error(status):
    if status == cv2.Stitcher_ERR_NEED_MORE_IMGS:
        logger.warning("[Stitcher error] 더 많은 이미지가 필요합니다.")
    elif status == cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL:
        logger.warning("[Stitcher error] 호모그래피 추정 실패. 이미지가 너무 다릅니다.")
    elif status == cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL:
        logger.warning("[Stitcher error] 카메라 파라미터 조정 실패.")
    else:
        logger.error("[Stitcher error] 알 수 없는 오류 발생")

# SRCNN 모델 정의
# To-do-First  Modularize first

class SRCNN(nn.Module):
    def __init__(self) -> None:
        super(SRCNN, self).__init__()
        # Feature extraction layer.
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, (9, 9), (1, 1), (4, 4)),
            nn.ReLU(True)
        )

        # Non-linear mapping layer.
        self.map = nn.Sequential(
            nn.Conv2d(64, 32, (5, 5), (1, 1), (2, 2)),
            nn.ReLU(True)
        )

        # Rebuild the layer.
        self.reconstruction = nn.Conv2d(32, 1, (5, 5), (1, 1), (2, 2))

        # Initialize model weights.
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    # Support torch.script function.
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = self.map(out)
        out = self.reconstruction(out)

        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight.data, 0.0, (2 / (module.out_channels * module.weight.data[0][0].numel())) ** 0.5)
                nn.init.zeros_(module.bias.data)

        nn.init.normal_(self.reconstruction.weight.data, 0.0, 0.001)
        nn.init.zeros_(self.reconstruction.bias.data)

#4layer 기반 SRCNN 모델 정의
class SRCNN_4Layer(nn.Module):
    def __init__(self) -> None:
        super(SRCNN, self).__init__()
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

        # Initialize model weights.
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    # Support torch.script function.
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = self.map(out)
        out = self.reconstruction(out)

        return out

    # The filter weight of each layer is a Gaussian distribution with zero mean and
    # standard deviation initialized by random extraction 0.001 (deviation is 0)
    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight.data, 0.0, math.sqrt(2 / (module.out_channels * module.weight.data[0][0].numel())))
                nn.init.zeros_(module.bias.data)

        nn.init.normal_(self.reconstruction.weight.data, 0.0, 0.001)
        nn.init.zeros_(self.reconstruction.bias.data)



# 이미지 향상 함수 (PyTorch SRCNN 사용)
# SRCNN 특성상 이미지 향상 시킨후 필터 투입
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

# 이미지 로드 및 전처리 함수
def load_and_preprocess_image(filepath, target_size=1000):
    try:
        img = cv2.imread(filepath)
        if img is not None:
            img = resize_image(img, target_size)
            img = cv2.GaussianBlur(img, (5, 5), 0)
        return img
    except Exception as e:
        logger.error(f"[preprocess] 이미지 로드 및 전처리 중 오류 발생: {e}")
        return None

# 동영상에서 프레임 추출 및 저장 함수
def extract_frames(video_path, output_dir, interval=1, enhance=False):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"[cv2 Error] 동영상 파일을 열 수 없습니다: {video_path}")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * interval)
        create_directory(output_dir)
        
        # SRCNN 모델 로드
        model = None
        if enhance:
            model = SRCNN()
            checkpoint = torch.load(MODEL_PATH, map_location=lambda storage, loc: storage, weights_only=True)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            model = model.to(memory_format=torch.channels_last, device=device)
        
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
        
        logger.info(f"[OpenCV] 프레임 {extracted_count}개를 {output_dir}에 저장하였습니다.")
        
    except Exception as e:
        logger.error(f"[OpenCV] 프레임 추출 중 오류 발생: {e}")
        raise

# 스티칭 후 디렉토리 정리 함수
def clean_up(directory):
    try:
        if os.path.isdir(directory):
            shutil.rmtree(directory)
    except Exception as e:
        logger.error(f"[CleanUP] 정리 중 오류 발생: {e}")

# API 리소스 정의
class Index(Resource):
    def get(self):
        return {'name': 'hongik', 'type': 'university'}

class Convert(Resource):
    def post(self):
        unique_id = str(uuid.uuid4())
        try:
            create_directory(UPLOAD_FOLDER)

            processing_dir = os.path.join(PROCESSING_FOLDER, unique_id)
            frames_dir = os.path.join(processing_dir, 'frames')
            create_directory(processing_dir)
            create_directory(frames_dir)

            result_dir = os.path.join(UPLOAD_FOLDER, unique_id)
            create_directory(result_dir)
            
            parser = reqparse.RequestParser()
            parser.add_argument('file', type=werkzeug.datastructures.FileStorage, location='files')
            args = parser.parse_args()
            file_object = args['file']

            if file_object is None:
                return jsonify({'result': 'failed', 'message': 'No file uploaded'})
            
            file_name = secure_filename(file_object.filename)
            video_path = os.path.join(processing_dir, file_name)
            file_object.save(video_path)
                
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
                    
                clean_up(processing_dir)
                    
                return send_file(result_path)
            else:
                clean_up(processing_dir)
                    
                return jsonify({'result': 'failed', 'message': 'No images to stitch'})
        except Exception as e:
            clean_up(processing_dir)
            logger.error(f"[Stitcher] Error during processing: {e}")
            return jsonify({'result': 'failed', 'message': str(e)})

# Add resources to API
api.add_resource(Index, '/')
api.add_resource(Convert, '/convert')

if __name__ == '__main__':
    application.run(debug=False, host='0.0.0.0', port=PORT)
