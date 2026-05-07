"""
로컬 전용 변환 스크립트 — 배포에 포함되지 않음.
실행 전: pip install ultralytics
사용법:  python convert_to_onnx.py
"""

from pathlib import Path
from ultralytics import YOLO

SRC = Path("runs/plogging_yolo11s_real_last/weights/best.pt")
DST_DIR = Path("model")
DST_DIR.mkdir(exist_ok=True)

print(f"변환 중: {SRC} ...")
model = YOLO(str(SRC))
model.export(format="onnx", imgsz=640, simplify=True, opset=12, dynamic=False)

exported = SRC.with_suffix(".onnx")
dst = DST_DIR / "best.onnx"
exported.rename(dst)

size_mb = round(dst.stat().st_size / 1e6, 1)
print(f"완료: {dst}  ({size_mb} MB)")
print("다음 단계: git add model/best.onnx && git commit -m 'feat: add ONNX model'")
