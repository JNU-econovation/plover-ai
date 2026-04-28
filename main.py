from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from detector import run_inference, run_inference_with_viz
from schemas import PredictResponse, VisualizeResponse

app = FastAPI(title="Plogging AI Detector")
app.mount("/static", StaticFiles(directory="static"), name="static")

_DASHBOARD_HTML = Path("static/index.html").read_text(encoding="utf-8")


async def _read_image(file: UploadFile) -> bytes:
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="빈 파일입니다.")
    return contents


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return HTMLResponse(content=_DASHBOARD_HTML)


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    """감지 결과만 반환. 백엔드 서버 간 통신용."""
    contents = await _read_image(file)
    try:
        return run_inference(contents)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"추론 중 오류 발생: {e}")


@app.post("/predict/visualize", response_model=VisualizeResponse)
async def predict_visualize(file: UploadFile = File(...)):
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
