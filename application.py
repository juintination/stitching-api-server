from flask import Flask, jsonify, send_file
from flask_restful import Resource, Api, reqparse
from flask_cors import CORS
import os, cv2
import werkzeug, uuid, shutil
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.layers import Input

application = Flask(__name__)
api = Api(application)
CORS(application)
application.config['JSON_AS_ASCII'] = False

# 경로 생성 함수
def create_directory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError as e:
        print(f"[mkdir Error] Failed to create the directory {directory}: {e}")
        raise
    
# 기존 스티칭 이미지 조절 함수
def resize_image(img, width=None, height=None):
    if width is None and height is None:
        return img
    else:
        h, w = img.shape[:2]
        if width is not None and height is not None:
            new_size = (width, height)
        elif width is not None:
            new_size = (width, int(h * (width / w)))
        else:
            new_size = (int(w * (height / h)), height)
        resized_img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)
        return resized_img

# 기존 이미지 스티칭 트리거 함수
def stitch_images(images, output_path):
    stitcher = cv2.Stitcher_create()
    status, stitched_img = stitcher.stitch(images)
    if status == cv2.Stitcher_OK:
        h, w = stitched_img.shape[:2]
        new_width = w
        new_height = int(w / 2)
        if new_height > h:
            new_height = h
            new_width = int(h * 2)
        resized_stitched_img = resize_image(stitched_img, new_width, new_height)
        cv2.imwrite(output_path, resized_stitched_img)
        print("[Stitch] Stitching Completed :", status)
    else:
        print("[Stitch Error] Stitching failed :", status)

# 기존 스티칭 과정 함수
def load_and_resize_image(filepath, width=None, height=None):
    img = cv2.imread(filepath)
    if img is not None:
        img = resize_image(img, width, height)
    return img

# 동영상에서 프레임 단위로 Shot 내려서 저장하는 함수
def extract_frames(video_path, output_dir, interval=0.6, enhance=False):
    try:
        '''
        def load_srcnn_model():
            input_shape = (None, None, 1)
            inputs = Input(shape=input_shape)
            conv1 = tf.keras.layers.Conv2D(128, (9, 9), activation='relu', padding='same')(inputs)
            conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
            outputs = tf.keras.layers.Conv2D(1, (5, 5), padding='same')(conv2)
            model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
            model.load_weights('./model/srcnn_weights.h5')
            return model

        def enhance_image(image, model):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(image)
            y = y.astype(np.float32) / 255.0
            y = np.expand_dims(np.expand_dims(y, axis=0), axis=-1)
            y_enhanced = model.predict(y)
            y_enhanced = y_enhanced[0, :, :, 0]
            y_enhanced = (y_enhanced * 255.0).clip(0, 255).astype(np.uint8)
            enhanced_image = cv2.merge([y_enhanced, cr, cb])
            enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_YCrCb2BGR)
            return enhanced_image
        '''

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[cv2 Error] Can not opening video file: {video_path}")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * interval)
        create_directory(output_dir)
        # srcnn_model = load_srcnn_model() if enhance else None
        frame_count = 0
        extracted_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                # if enhance and srcnn_model:
                #     frame = enhance_image(frame, srcnn_model)
                frame_filename = os.path.join(output_dir, f"frame_{extracted_count:04d}.jpg")
                cv2.imwrite(frame_filename, frame)
                extracted_count += 1
            frame_count += 1
        cap.release()
        
        print(f"[Message] Extracted {extracted_count} frames to {output_dir}")
        
    except Exception as e:
        
        print(f"[Error] Occurred during frame extraction: {e}")
        raise

# 데이터 프로세싱 및 스티칭 후 비우는 함수
def clean_up(directory):
    try:
        if os.path.isdir(directory):
            shutil.rmtree(directory)
            
    except Exception as e:
        print(f"[Remove Error] Occurred during cleanup: {e}")

class Index(Resource):
    def get(self):
        return {'name': 'juintination',
                'email': 'juintin@kakao.com'}

class Convert(Resource):
    def post(self):
        unique_id = str(uuid.uuid4())
        try:
            create_directory('uploads')

            processing_dir = os.path.join('processing', unique_id)
            frames_dir = os.path.join(processing_dir, 'frames')
            create_directory(processing_dir)
            create_directory(frames_dir)

            result_dir = os.path.join('uploads', unique_id)
            create_directory(result_dir)
            
            parser = reqparse.RequestParser()
            parser.add_argument('file', type=werkzeug.datastructures.FileStorage, location='files')
            args = parser.parse_args()
            file_object = args['file']

            if file_object is None:
                return jsonify({'result': 'failed', 'message': 'No file uploaded'})
            if result_dir is None:
                return jsonify({'result': 'failed', 'message': 'No output directory specified'})
            
            video_path = os.path.join(processing_dir, file_object.filename)
            file_object.save(video_path)
                
            extract_frames(video_path, frames_dir, interval=0.6, enhance=True)
                
            images = []
            for img_file in sorted(os.listdir(frames_dir)):
                img_path = os.path.join(frames_dir, img_file)
                img = load_and_resize_image(img_path, width=2000)
                if img is not None:
                    images.append(img)
                
            result_path = os.path.join(result_dir, 'result_img.jpg')
            if len(images) > 0:
                print('[Message] Start Stitching images')
                stitch_images(images, result_path)
                    
                # Clean up after processing
                clean_up(processing_dir)
                    
                return send_file(result_path)
            else:
                # Clean up after processing, even if no images were stitched
                clean_up(processing_dir)
                    
                return jsonify({'result': 'failed', 'message': 'No images to stitch'})
        except Exception as e:
            # Clean up in case of an error
            clean_up(processing_dir)
            print(f"[Error] Occurred during processing: {e}")
            return jsonify({'result': 'failed', 'message': str(e)})

api.add_resource(Index, '/')
api.add_resource(Convert, '/convert')

if __name__ == '__main__':
    application.run(debug=True, host='0.0.0.0', port=5050)
