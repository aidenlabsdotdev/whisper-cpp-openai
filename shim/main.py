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
        return JSONResponse(content=r.json())
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
