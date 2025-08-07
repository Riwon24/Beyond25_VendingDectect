import onnxruntime
import cv2
import numpy as np

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    original_img = img.copy() # 원본 이미지
    img = cv2.resize(img, (640, 640)) # YOLOv8 기본 입력 크기
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0 # 0~1 정규화
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0) # 배치 차원 추가 (1,3,640,640)
    return img, original_img

def run_inference(session, input_tensor):
    outputs = session.run(None, {"images": input_tensor})
    return outputs

''' 
utils.py 함수 사용 예시 코드
# ONNX 모델 로드
session = onnxruntime.InferenceSession("best.onnx")
# 이미지 전처리
input_tensor, original_img = preprocess_image("test_images/test.jpg")
# ONNX 추론 실행
output = run_inference(session, input_tensor)
'''