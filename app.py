# A,B단계에서 가져올 부분
from utils import preprocess_image, run_inference
from visualize import extract_detected_items, draw_boxes

import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import io

# 사진 업로드 하는 화면 설정
st.title("Upload Image!")
st.text('')

# 파일 업로드
uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    input_tensor = preprocess_image(img) #A; 이미지를 입력텐서로 변환
    output = run_inference(session, input_tensor) #A; 모델 추론 실행 함수
    detected_items = extract_detected_items(output) #B; 결과에서 품목 추출
    result_img = draw_boxes(np.array(img), output) #B; 이미지 위에 박스그릠

    st.image(result_img, caption="탐지 결과")
    st.write("탐지된 품목:", detected_items)
