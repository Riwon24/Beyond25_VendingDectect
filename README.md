# 🚩 Beyond-SummerProject-Vending Item Detector
2025 Beyond 여름 프로젝트-자판기 물품 탐지기

## 📍프로그램 설명
YOLOv8 모델을 이용하여 자판기 이미지를 삽입하면
ONNX 추론을 실행, 품목 리스트 + 이미지 시각화를 거쳐 Streamlit UI로 자판기 안의 아이템 목록을 출력합니다.

## 📍실행 방법
1. requirements.txt의 패키지 설치
2. best.onnx, classes.txt, utils.py, visualize.py, app.py를 같은 디렉토리에 두기
3. CMD에서 2번 디렉토리로 이동>'streamlit run app.py' 실행

4. 파일 업로드(jpg, png, jpeg)
5. 결과 출력

## 📍메인 코드 동작 흐름 설명
### 1. 라이브러리 및 함수 불러오기
-'requirements.txt' 패키지 및 'utils.py', 'visualize.py' 함수 불러오기

### 2. ONNX 모델 로드
**session = onnxruntime.InferenceSession("best.onnx")**

### 3. 업로드 된 이미지 처리 함수
**uploaded(uploaded_file, session)**

- streamlit로 읽은 파일을 OpenCV 이미지로 디코딩
- 전처리에서 변환된 입력텐서를 받아와 모델 추론
- 'classes.txt'에서 클래스 이름 읽어옴
- 출력 후처리 및 결과에서 품목 추출
- 박스 그림 이미지 출력

### 4. UI에서 출력
- 제목 텍스트 및 파일 업로드 버튼
-> 이미지 업로드
- 3번의 이미지 처리 함수 실행
- 탐지된 아이템 박스 이미지 및 아이템 목록 결과 출력

### 📍각 py 파일 함수 설명
- 'utils.py'
preprocess_image(img)
: 이미지 전처리(이미지를 YOLO 형식 입력 텐서로 변환)
run_inference(session, input_tensor)
: ONNX 모델 추론 실행

- 'visualize.py'
yolov8_postprocess(output, conf_threshold, iou_threshold)
: 모델 출력 후처리
extract_detected_items(output, class_names)
: ONNX 결과에서 품목 추출
scale_boxes(boxes, input_shape, orig_shape)
: 원본 이미지 해상도로 박스 크기 변환
draw_boxes(image, output, class_names)
: 이미지 위에 박스 + 텍스트 표시
