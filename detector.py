import base64
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from schemas import DetectionResult, PredictResponse, VisualizeResponse

# 클래스 매핑
CLASS_NAMES = {
    0: "담배꽁초", 1: "가구류", 2: "도기류", 3: "비닐류", 4: "스티로폼류",
    5: "유리병류", 6: "의류", 7: "자전거", 8: "전자제품", 9: "종이류",
    10: "캔류", 11: "페트병류", 12: "플라스틱류",
}

_PALETTE = [
    (0, 220, 100), (220, 50, 50), (50, 100, 220), (220, 200, 0), (0, 200, 220),
    (200, 0, 200), (100, 220, 50), (220, 130, 0), (0, 130, 220), (130, 0, 220),
    (220, 130, 130), (130, 220, 130), (130, 130, 220),
]

DEVICE = 0 if torch.cuda.is_available() else "cpu"
model = YOLO("runs/plogging_yolo11s_real_last/weights/best.pt")

def _preprocess_image(img: np.ndarray) -> np.ndarray:
    """투명 객체 탐지율 향상을 위한 전처리: 대비 강화(CLAHE)"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def _draw_boxes(image: np.ndarray, boxes) -> np.ndarray:
    annotated = image.copy()
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        name = CLASS_NAMES.get(cls_id, "알 수 없음")
        color = _PALETTE[cls_id % len(_PALETTE)]
        label = f"{name} {conf:.2f}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3) # 박스 두께를 살짝 키움
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 12), (x1 + tw, y1), color, -1)
        cv2.putText(annotated, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return annotated

def _predict(image_bytes: bytes) -> tuple:
    """공통 추론 로직. (원본 img, boxes, PredictResponse 기본 필드) 반환."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("이미지를 읽을 수 없습니다.")

    processed_img = _preprocess_image(img)

    start = time.perf_counter()
    results = model.predict(
        processed_img,
        conf=0.25,
        iou=0.45,
        imgsz=640,
        augment=False,
        device=DEVICE,
        verbose=False,
    )
    inference_ms = round((time.perf_counter() - start) * 1000, 2)

    boxes = results[0].boxes
    counts: dict[str, int] = {}
    detections: list[DetectionResult] = []

    for box in boxes:
        cls_id = int(box.cls[0])
        name = CLASS_NAMES.get(cls_id, "알 수 없음")
        if name == "전자제품":
            continue
        conf = float(box.conf[0])
        counts[name] = counts.get(name, 0) + 1
        detections.append(DetectionResult(type=name, confidence=round(conf, 4)))

    return img, boxes, counts, detections, inference_ms


def run_inference(image_bytes: bytes) -> PredictResponse:
    """/predict 전용 — 감지 결과만 반환 (이미지 없음)."""
    _, _boxes, counts, detections, inference_ms = _predict(image_bytes)
    return PredictResponse(
        status="success",
        total_count=len(_boxes),
        counts=counts,
        detections=detections,
        inference_speed_ms=inference_ms,
    )


def run_inference_with_viz(image_bytes: bytes) -> VisualizeResponse:
    """/predict/visualize 전용 — 감지 결과 + 박스 이미지(Base64) 반환."""
    img, boxes, counts, detections, inference_ms = _predict(image_bytes)
    annotated = _draw_boxes(img, boxes)
    _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    encoded = base64.b64encode(buffer).decode("utf-8")
    return VisualizeResponse(
        status="success",
        total_count=len(boxes),
        counts=counts,
        detections=detections,
        inference_speed_ms=inference_ms,
        annotated_image_base64=encoded,
    )