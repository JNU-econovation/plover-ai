from pydantic import BaseModel
from typing import Dict, List


class DetectionResult(BaseModel):
    type: str
    confidence: float


class PredictResponse(BaseModel):
    status: str
    total_count: int
    counts: Dict[str, int]
    detections: List[DetectionResult]
    inference_speed_ms: float


class VisualizeResponse(PredictResponse):
    annotated_image_base64: str
