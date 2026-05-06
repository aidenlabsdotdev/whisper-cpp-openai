import os
import time
from typing import Optional

import httpx
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

WHISPER_URL = os.getenv("WHISPER_URL", "http://whisper:8080").rstrip("/")
MODEL_NAME = os.getenv("WHISPER_MODEL_NAME", "whisper-1")
API_KEY = os.getenv("SHIM_API_KEY") or None
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "600"))

app = FastAPI(title="whisper.cpp OpenAI shim")


def check_auth(authorization: Optional[str] = Header(default=None)) -> None:
    if API_KEY is None:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    if authorization.removeprefix("Bearer ").strip() != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/health")
async def health() -> dict:
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{WHISPER_URL}/")
        upstream = "ok" if r.status_code < 500 else "degraded"
    except Exception:
        upstream = "down"
    return {"status": "ok", "upstream": upstream}


@app.get("/v1/models", dependencies=[Depends(check_auth)])
async def list_models() -> dict:
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "whisper.cpp",
            }
        ],
    }


_SEGMENT_DEFAULTS = {
    "seek": 0,
    "temperature": 0.0,
    "avg_logprob": 0.0,
    "compression_ratio": 1.0,
    "no_speech_prob": 0.0,
}


def _normalize_verbose_json(payload: dict) -> dict:
    """Reshape whisper.cpp's verbose_json into OpenAI's verbose_json schema.

    whisper.cpp nests word timestamps inside each segment and omits several
    segment-level metrics; OpenAI exposes words at the top level (with a
    smaller field set) and requires seek/temperature/avg_logprob/etc.
    """
    top_words: list[dict] = []
    segments_in = payload.get("segments") or []
    segments_out: list[dict] = []
    for seg in segments_in:
        normalized = {**_SEGMENT_DEFAULTS, **seg}
        for w in normalized.pop("words", []) or []:
            top_words.append(
                {"word": w.get("word", ""), "start": w.get("start"), "end": w.get("end")}
            )
        segments_out.append(normalized)
    payload["segments"] = segments_out
    if top_words:
        payload["words"] = top_words
    return payload


async def _proxy_inference(
    file: UploadFile,
    response_format: str,
    temperature: float,
    language: Optional[str],
    prompt: Optional[str],
    translate: bool,
) -> Response:
    data: dict[str, str] = {
        "response_format": response_format,
        "temperature": str(temperature),
    }
    if language and language != "auto":
        data["language"] = language
    if prompt:
        data["prompt"] = prompt
    if translate:
        data["translate"] = "true"

    payload = await file.read()
    files = {
        "file": (
            file.filename or "audio",
            payload,
            file.content_type or "application/octet-stream",
        )
    }

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        r = await client.post(f"{WHISPER_URL}/inference", data=data, files=files)

    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail=r.text)

    ct = r.headers.get("content-type", "application/json")
    if "json" in ct:
        body = r.json()
        if response_format == "verbose_json" and isinstance(body, dict):
            body = _normalize_verbose_json(body)
        return JSONResponse(content=body)
    return Response(content=r.content, media_type=ct)


@app.post("/v1/audio/transcriptions", dependencies=[Depends(check_auth)])
async def transcriptions(
    file: UploadFile = File(...),
    model: str = Form(default=MODEL_NAME),
    language: Optional[str] = Form(default=None),
    prompt: Optional[str] = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0),
) -> Response:
    return await _proxy_inference(
        file=file,
        response_format=response_format,
        temperature=temperature,
        language=language,
        prompt=prompt,
        translate=False,
    )


@app.post("/v1/audio/translations", dependencies=[Depends(check_auth)])
async def translations(
    file: UploadFile = File(...),
    model: str = Form(default=MODEL_NAME),
    prompt: Optional[str] = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0),
) -> Response:
    return await _proxy_inference(
        file=file,
        response_format=response_format,
        temperature=temperature,
        language=None,
        prompt=prompt,
        translate=True,
    )
