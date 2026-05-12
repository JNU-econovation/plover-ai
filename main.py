import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from detector import run_inference, run_inference_with_viz
from schemas import PredictResponse, VisualizeResponse

load_dotenv()

_API_KEY = os.getenv("API_KEY")

# GET /predict* 봇 차단 설정
_VIOLATION_WINDOW = 60  # 위반 카운트 집계 시간 (초)
_MAX_VIOLATIONS = 5  # 이 횟수 초과 시 IP 차단
_BAN_DURATION = 3600  # 차단 유지 시간 (초, 1시간)

_violations: dict[str, list[float]] = defaultdict(list)
_banned_ips: dict[str, float] = {}

app = FastAPI(title="Plogging AI Detector")


@app.middleware("http")
async def block_predict_get_bots(request: Request, call_next):
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()

    # 차단된 IP 확인
    ban_until = _banned_ips.get(client_ip)
    if ban_until:
        if now < ban_until:
            return JSONResponse(
                status_code=403, content={"detail": "접근이 차단되었습니다."}
            )
        del _banned_ips[client_ip]

    # GET /predict* 요청 감지
    if request.method == "GET" and request.url.path.startswith("/predict"):
        recent = [t for t in _violations[client_ip] if now - t < _VIOLATION_WINDOW]
        recent.append(now)
        _violations[client_ip] = recent

        if len(recent) >= _MAX_VIOLATIONS:
            _banned_ips[client_ip] = now + _BAN_DURATION
            del _violations[client_ip]
            return JSONResponse(
                status_code=403, content={"detail": "접근이 차단되었습니다."}
            )

        return JSONResponse(
            status_code=405, content={"detail": "허용되지 않는 메소드입니다."}
        )

    return await call_next(request)


app.mount("/static", StaticFiles(directory="static"), name="static")

_DASHBOARD_HTML = Path("static/index.html").read_text(encoding="utf-8")


async def verify_api_key(x_api_key: Optional[str] = Header(None)) -> None:
    if not _API_KEY:
        return
    if x_api_key != _API_KEY:
        raise HTTPException(status_code=401, detail="유효하지 않은 API 키입니다.")


async def _read_image(file: UploadFile) -> bytes:
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="빈 파일입니다.")
    return contents


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return HTMLResponse(content=_DASHBOARD_HTML)


@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    _: None = Depends(verify_api_key),
):
    """감지 결과만 반환. 백엔드 서버 간 통신용."""
    contents = await _read_image(file)
    try:
        result = run_inference(contents)
        result.latitude = latitude
        result.longitude = longitude
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"추론 중 오류 발생: {e}")


@app.post("/predict/visualize", response_model=VisualizeResponse)
async def predict_visualize(
    file: UploadFile = File(...),
    _: None = Depends(verify_api_key),
):
    """감지 결과 + 박스 이미지(Base64) 반환. 웹 대시보드 전용."""
    contents = await _read_image(file)
    try:
        return run_inference_with_viz(contents)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"추론 중 오류 발생: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
