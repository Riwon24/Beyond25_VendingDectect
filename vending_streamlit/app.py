# A,B단계에서 가져올 부분
from utils import preprocess_image, run_inference
from visualize import yolov8_postprocess, extract_detected_items, draw_boxes

import streamlit as st
from PIL import Image
import numpy as np
import cv2

# 모델 관련 불러오기
import onnxruntime as ort
session = ort.InferenceSession("best.onnx")

def uploaded(uploaded_file, session):
    # 바이트로 읽은 파일을 numpy 배열로 디코딩(cv2가 읽을 수 있게)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # A에서 변환된 입력텐서랑 원본이미지 둘 다 받아옴(둘다 안받아오면 오류)
    input_tensor, original_img = preprocess_image(img)
    output = run_inference(session, input_tensor)  # A; 모델 추론 실행

    # 후처리
    boxes = yolov8_postprocess(output)

    with open("classes.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines()]  # classes.txt에서 클래스 읽어옴

    detected_items = extract_detected_items(boxes, class_names)  # B; 결과에서 품목 추출
    result_img = draw_boxes(img, boxes, classes_names)  # B; 이미지 위에 박스그릠

    return detected_items, result_img, boxes

# ---제목 및 업로드 UI
st.title("Welcome to Vending Item Detector!")

# 파일 업로드(파일내용 자체가 바이트로 streamlit안에 저장)
uploaded_file = st.file_uploader("Upload an image for item detection using an ONNX model of YOLOv8", type=["jpg", "png"])

# ---파일이 입력되었을 때
if uploaded_file:
    detected_items, result_img, boxes = uploaded(uploaded_file, session)
    st.image(result_img, caption="탐지 결과")
    st.write("탐지된 품목:", detected_items)

    # 디버그
    st.write('모델 입력 텐서 이름: ' ,session.get_inputs()[0].name)
    st.write("후처리 결과 박스 수:", len(boxes))
