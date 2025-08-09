# Beyond-SummerProject-Vending Item Detector
2025 Beyond 여름 프로젝트-자판기 물품 탐지기

## 프로젝트 구조
vending_streamlit/

├── app.py              ← streamlit 실행 파일

├── requirements.txt    ← 전체 프로그램에 필요한 패키지 목록 txt 파일

├── README.md           ← 프로그램 설명 README 파일

├── best.onnx           ← YOLOv8로 학습된 모델을 ONNX로 변환한 모델 파일

├── classes.txt         ← 클래스 이름 목록 txt 파일

├── utils.py            ← 전처리 및 추론 함수 py 파일

├── visualize.py        ← 후처리 및 박스 출력 함수 py 파일

├── test_images/        ← 전처리 테스트 예시 이미지(optional)

│   └── test1.jpg

├── result_demo.jpg    ← 후처리 결과 예시 이미지(optional)
