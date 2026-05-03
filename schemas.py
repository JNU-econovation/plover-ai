from pydantic import BaseModel
from typing import Dict, List, Optional


class DetectionResult(BaseModel):
    type: str
    confidence: float


class PredictResponse(BaseModel):
    status: str
    total_count: int
    counts: Dict[str, int]
    detections: List[DetectionResult]
    inference_speed_ms: float
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class VisualizeResponse(PredictResponse):
    annotated_image_base64: str
