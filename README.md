# Plogging AI Detector

플로버 AI 서버입니다.
사진을 업로드하면 YOLOv11 모델이 쓰레기를 감지하고 종류와 개수를 반환합니다.

---

## 기술 스택

- **FastAPI** — REST API 서버
- **YOLOv11s** (ultralytics) — 객체 감지 모델
- **OpenCV** — 이미지 전처리 및 시각화
- **PyTorch** — 딥러닝 런타임 (GPU/CPU 자동 선택)

---

## 실행 방법

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 모델 가중치 배치

모델 가중치 파일(`best.pt`)은 Git에 포함되지 않습니다. 아래 경로에 직접 배치해주세요.

```
runs/plogging_yolo11s_real_last/weights/best.pt
```

### 3. 서버 실행

```bash
# 개발 (코드 변경 시 자동 재시작)
uvicorn main:app --reload

# 배포
uvicorn main:app --host 0.0.0.0 --port 8000
```

서버가 실행되면 `http://localhost:8000` 에서 웹 대시보드를 확인할 수 있습니다.

---

## API

### `POST /predict`

이미지를 분석하여 쓰레기 감지 결과를 반환합니다. 프론트엔드 및 백엔드 서버 간 통신에 사용합니다.

**Request** — `multipart/form-data`

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `file` | File | ✅ | 분석할 이미지 (JPEG, PNG 등) |
| `latitude` | float | | 사진 촬영 위도 |
| `longitude` | float | | 사진 촬영 경도 |

**Response**

```json
{
  "status": "success",
  "total_count": 4,
  "counts": { "담배꽁초": 2, "캔류": 1, "비닐류": 1 },
  "detections": [
    { "type": "담배꽁초", "confidence": 0.9123 },
    { "type": "캔류", "confidence": 0.7654 }
  ],
  "inference_speed_ms": 213.45,
  "latitude": 35.1595,
  "longitude": 126.8526
}
```

**예시 (curl)**

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@photo.jpg" \
  -F "latitude=35.1595" \
  -F "longitude=126.8526"
```

**예시 (JavaScript)**

```javascript
const formData = new FormData();
formData.append("file", imageFile);
formData.append("latitude", 35.1595);
formData.append("longitude", 126.8526);

const res = await fetch("http://localhost:8000/predict", {
  method: "POST",
  body: formData,
});
const data = await res.json();
```

---

### `POST /predict/visualize`

감지 결과에 더해 바운딩 박스가 그려진 이미지를 Base64로 함께 반환합니다.
웹 대시보드(`GET /`) 전용 엔드포인트입니다.

`/predict` 응답과 동일하며 아래 필드가 추가됩니다.

| 필드 | 타입 | 설명 |
|------|------|------|
| `annotated_image_base64` | string | 바운딩 박스 이미지 (Base64 JPEG) |

```html
<img src="data:image/jpeg;base64,{{ annotated_image_base64 }}" />
```

---

### 에러 응답

| 상태 코드 | 원인 |
|-----------|------|
| `400` | 이미지가 아닌 파일 또는 빈 파일 |
| `422` | 이미지 디코딩 실패 (손상된 파일) |
| `500` | 서버 내부 오류 |

```json
{ "detail": "이미지 파일만 업로드 가능합니다." }
```

---

## 감지 가능한 쓰레기 종류

`담배꽁초` `가구류` `도기류` `비닐류` `스티로폼류` `유리병류` `의류` `자전거` `종이류` `캔류` `페트병류` `플라스틱류`

> 전자제품은 모델이 감지하더라도 응답에서 제외됩니다.

---

## API 문서 (Swagger UI)

서버 실행 후 아래 주소에서 대화형 API 문서를 확인할 수 있습니다.

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
