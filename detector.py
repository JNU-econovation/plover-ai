import base64
import time
import cv2
import numpy as np
import onnxruntime as ort
from schemas import DetectionResult, PredictResponse, VisualizeResponse

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

_IMG_SIZE = 640
_session = ort.InferenceSession("model/best.onnx", providers=["CPUExecutionProvider"])
_input_name = _session.get_inputs()[0].name


def _preprocess(img: np.ndarray) -> tuple[np.ndarray, float, tuple[float, float]]:
    """CLAHE + letterbox → (NCHW tensor, scale_ratio, (pad_w, pad_h))"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img = cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)

    h, w = img.shape[:2]
    r = min(_IMG_SIZE / h, _IMG_SIZE / w)
    new_w, new_h = round(w * r), round(h * r)
    dw = (_IMG_SIZE - new_w) / 2
    dh = (_IMG_SIZE - new_h) / 2

    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(
        img,
        int(round(dh - 0.1)), int(round(dh + 0.1)),
        int(round(dw - 0.1)), int(round(dw + 0.1)),
        cv2.BORDER_CONSTANT, value=(114, 114, 114),
    )

    tensor = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))[np.newaxis]  # (1, 3, 640, 640)
    return tensor, r, (dw, dh)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while len(order):
        i = int(order[0])
        keep.append(i)
        if len(order) == 1:
            break
        inter = (
            np.maximum(0, np.minimum(x2[i], x2[order[1:]]) - np.maximum(x1[i], x1[order[1:]]))
            * np.maximum(0, np.minimum(y2[i], y2[order[1:]]) - np.maximum(y1[i], y1[order[1:]]))
        )
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= iou_threshold]
    return keep


def _postprocess(
    raw: np.ndarray,
    orig_shape: tuple[int, int],
    ratio: float,
    pad: tuple[float, float],
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
) -> list[dict]:
    """ultralytics ONNX 출력 (1, 4+nc, 8400) → [{box, cls_id, conf}]"""
    pred = raw[0].T  # (8400, 4+nc)

    class_scores = pred[:, 4:]
    class_ids = class_scores.argmax(axis=1)
    confs = class_scores[np.arange(len(class_scores)), class_ids]

    mask = confs >= conf_thresh
    if not mask.any():
        return []

    pred, confs, class_ids = pred[mask], confs[mask], class_ids[mask]
    cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)

    keep = _nms(boxes, confs, iou_thresh)

    dw, dh = pad
    orig_h, orig_w = orig_shape
    results = []
    for i in keep:
        x1, y1, x2, y2 = boxes[i]
        x1 = int(max(0, min(round((x1 - dw) / ratio), orig_w)))
        y1 = int(max(0, min(round((y1 - dh) / ratio), orig_h)))
        x2 = int(max(0, min(round((x2 - dw) / ratio), orig_w)))
        y2 = int(max(0, min(round((y2 - dh) / ratio), orig_h)))
        results.append({"box": (x1, y1, x2, y2), "cls_id": int(class_ids[i]), "conf": float(confs[i])})
    return results


def _draw_boxes(image: np.ndarray, detections: list[dict]) -> np.ndarray:
    annotated = image.copy()
    for d in detections:
        cls_id, conf = d["cls_id"], d["conf"]
        x1, y1, x2, y2 = d["box"]
        name = CLASS_NAMES.get(cls_id, "알 수 없음")
        color = _PALETTE[cls_id % len(_PALETTE)]
        label = f"{name} {conf:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 12), (x1 + tw, y1), color, -1)
        cv2.putText(annotated, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return annotated


def _predict(image_bytes: bytes) -> tuple:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("이미지를 읽을 수 없습니다.")

    orig_shape = img.shape[:2]
    tensor, ratio, pad = _preprocess(img)

    start = time.perf_counter()
    outputs = _session.run(None, {_input_name: tensor})
    inference_ms = round((time.perf_counter() - start) * 1000, 2)

    raw_detections = _postprocess(outputs[0], orig_shape, ratio, pad)
    total_count = len(raw_detections)

    counts: dict[str, int] = {}
    det_results: list[DetectionResult] = []
    for d in raw_detections:
        name = CLASS_NAMES.get(d["cls_id"], "알 수 없음")
        if name == "전자제품":
            continue
        counts[name] = counts.get(name, 0) + 1
        det_results.append(DetectionResult(type=name, confidence=round(d["conf"], 4)))

    return img, raw_detections, total_count, counts, det_results, inference_ms


def run_inference(image_bytes: bytes) -> PredictResponse:
    _, _, total_count, counts, detections, inference_ms = _predict(image_bytes)
    return PredictResponse(
        status="success",
        total_count=total_count,
        counts=counts,
        detections=detections,
        inference_speed_ms=inference_ms,
    )


def run_inference_with_viz(image_bytes: bytes) -> VisualizeResponse:
    img, raw_detections, total_count, counts, detections, inference_ms = _predict(image_bytes)
    annotated = _draw_boxes(img, raw_detections)
    _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    encoded = base64.b64encode(buffer).decode("utf-8")
    return VisualizeResponse(
        status="success",
        total_count=total_count,
        counts=counts,
        detections=detections,
        inference_speed_ms=inference_ms,
        annotated_image_base64=encoded,
    )
