# view.py
import os
import cv2
import uuid
import shutil
import numpy as np
import torch
from torch import nn
from django.conf import settings
from django.http import JsonResponse, FileResponse
from django.views.decorators.csrf import csrf_exempt
from .models import SRCNN, SRCNN_4Layer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_directory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError as e:
        logger.error(f"[mkdir Error] 디렉토리 {directory} 생성 실패: {e}")
        raise


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


def stitch_images(images, output_path):
    cv2.ocl.setUseOpenCL(False)  # OpenCL 비활성화
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


def handle_stitching_error(status):
    if status == cv2.Stitcher_ERR_NEED_MORE_IMGS:
        logger.warning("[Stitcher error] 더 많은 이미지가 필요합니다.")
    elif status == cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL:
        logger.warning("[Stitcher error] 호모그래피 추정 실패. 이미지가 너무 다릅니다.")
    elif status == cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL:
        logger.warning("[Stitcher error] 카메라 파라미터 조정 실패.")
    else:
        logger.error("[Stitcher error] 알 수 없는 오류 발생")


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

def load_model(layer_choose):
    if layer_choose == 3:
        model = SRCNN()
        checkpoint = torch.load(settings.MODEL_PATH, map_location=lambda storage, loc: storage)
    elif layer_choose == 4:
        model = SRCNN_4Layer()
        checkpoint = torch.load(settings.MODEL_4LAYER_PATH, map_location=lambda storage, loc: storage)
    else:
        raise ValueError("Invalid model layer choice")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model.to(memory_format=torch.channels_last, device=device)

def extract_frames(video_path, output_dir, interval=1, enhance=False):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"[cv2 Error] 동영상 파일을 열 수 없습니다: {video_path}")
            return

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

        logger.info(f"[OpenCV] 프레임 {extracted_count}개를 {output_dir}에 저장하였습니다.")

    except Exception as e:
        logger.error(f"[OpenCV] 프레임 추출 중 오류 발생: {e}")
        raise


def clean_up(directory):
    try:
        if os.path.isdir(directory):
            shutil.rmtree(directory)
            logger.info(f"[CleanUP] {directory} 정리 완료")
    except Exception as e:
        logger.error(f"[CleanUP] 정리 중 오류 발생: {e}")


def index(request):
    return JsonResponse({'name': 'hongik', 'type': 'university'})


@csrf_exempt
def convert(request):
    if request.method == 'POST':
        unique_id = str(uuid.uuid4())
        processing_dir = os.path.join(settings.PROCESSING_FOLDER, unique_id)
        frames_dir = os.path.join(processing_dir, 'frames')
        result_dir = os.path.join(settings.UPLOAD_FOLDER, unique_id)
        try:
            create_directory(settings.UPLOAD_FOLDER)
            create_directory(processing_dir)
            create_directory(frames_dir)
            create_directory(result_dir)

            file = request.FILES.get('file')
            if not file:
                return JsonResponse({'result': 'failed', 'message': 'No file uploaded'})

            video_path = os.path.join(processing_dir, file.name)
            with open(video_path, 'wb+') as destination:
                for chunk in file.chunks():
                    destination.write(chunk)

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
                return FileResponse(open(result_path, 'rb'))
            else:
                clean_up(processing_dir)
                return JsonResponse({'result': 'failed', 'message': 'No images to stitch'})
        except Exception as e:
            clean_up(processing_dir)
            logger.error(f"[Stitcher] Error during processing: {e}")
            return JsonResponse({'result': 'failed', 'message': str(e)})

    return JsonResponse({'result': 'failed', 'message': 'Invalid request method'})