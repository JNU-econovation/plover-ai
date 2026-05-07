import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from detector import run_inference, run_inference_with_viz
from schemas import PredictResponse, VisualizeResponse

load_dotenv()

_API_KEY = os.getenv("API_KEY")

app = FastAPI(title="Plogging AI Detector")
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
