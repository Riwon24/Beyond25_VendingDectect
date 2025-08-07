import numpy as np
import cv2
from collections import Counter

# sigmoid 함수 정의
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# YOLOv8 후처리 함수 (ONNX용 + 디버깅 + NMS 지원)
def yolov8_postprocess(output, conf_threshold=0.5, iou_threshold=0.3, debug=False):
    """
    output: ONNX 모델 추론 결과, shape = (1, 10, 8400)
    return: boxes = [x1, y1, x2, y2, confidence, class_id]
    """
    preds = np.squeeze(output[0], axis=0)  # → (10, 8400)
    preds = preds.T  # → (8400, 10)

    boxes = preds[:, :4]                  # cx, cy, w, h
    raw_objectness = preds[:, 4]
    raw_class_scores = preds[:, 5:]

    objectness = sigmoid(raw_objectness)
    class_probs = sigmoid(raw_class_scores)

    class_ids = np.argmax(class_probs, axis=1)
    class_scores = np.max(class_probs, axis=1)
    confidences = objectness * class_scores

    # confidence threshold 필터링
    mask = confidences > conf_threshold
    boxes = boxes[mask]
    class_ids = class_ids[mask]
    confidences = confidences[mask]

    # 좌표 변환: cx, cy, w, h → x1, y1, x2, y2
    final_boxes = []
    for i in range(len(boxes)):
        cx, cy, w, h = boxes[i]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        final_boxes.append([x1, y1, x2, y2, confidences[i], class_ids[i]])

    final_boxes = np.array(final_boxes)

    # NMS 적용 (OpenCV 사용)
    if len(final_boxes) > 0:
        bboxes = final_boxes[:, :4].tolist()
        scores = final_boxes[:, 4].tolist()
        indices = cv2.dnn.NMSBoxes(bboxes, scores, conf_threshold, iou_threshold)
        if len(indices) > 0:
            final_boxes = final_boxes[indices.flatten()]
        else:
            final_boxes = np.empty((0, 6))  # No boxes remaining after NMS

    # 디버깅 출력
    if debug:
        print(f"🔍 ONNX 출력 총 box 후보: {preds.shape[0]}")
        print(f"✅ confidence threshold: {conf_threshold}")
        print(f"✅ passed boxes (before NMS): {len(mask.nonzero()[0])}")
        print(f"✅ final boxes (after NMS): {len(final_boxes)}")
        for i in range(min(10, len(final_boxes))):
            x1, y1, x2, y2, conf, cls = final_boxes[i]
            print(f"[{i}] conf={conf:.3f}, class={int(cls)}")

        # 클래스별 박스 통계 출력
        cls_ids = [int(cls) for *_, cls in final_boxes]
        print("✅ 클래스별 탐지 결과 분포:", Counter(cls_ids))

    return final_boxes

# 원본 이미지 크기에 맞게 box 좌표 scale
def scale_boxes(boxes, input_shape, orig_shape):
    gain_w = orig_shape[1] / input_shape[1]
    gain_h = orig_shape[0] / input_shape[0]
    scaled = []
    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        x1 *= gain_w
        x2 *= gain_w
        y1 *= gain_h
        y2 *= gain_h
        scaled.append([x1, y1, x2, y2, box[4], box[5]])
    return np.array(scaled)

# 탐지된 클래스 이름 리스트 반환
def extract_detected_items(boxes, class_names, conf_threshold=0.5):
    detected = []
    for box in boxes:
        conf = box[4]
        cls_id = int(box[5])
        if conf >= conf_threshold and 0 <= cls_id < len(class_names):
            detected.append(class_names[cls_id])
    return list(set(detected))

# 이미지에 box 및 라벨 시각화
def draw_boxes(img, boxes, class_names, conf_threshold=0.5):
    img_copy = img.copy()
    for box in boxes:
        x1, y1, x2, y2 = [int(b) for b in box[:4]]
        conf = box[4]
        cls_id = int(box[5])
        if conf < conf_threshold:
            continue
        label = f"{class_names[cls_id]} {conf:.2f}"  # ✅ conf 포함 출력
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img_copy, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
    return img_copy
