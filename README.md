# Beyond25_VendingDetect

''' 
utils.py 함수 사용 예시 코드
# ONNX 모델 로드
session = onnxruntime.InferenceSession("best.onnx")
# 이미지 전처리
input_tensor, original_img = preprocess_image("test_images/test.jpg")
# ONNX 추론 실행
output = run_inference(session, input_tensor)
'''
