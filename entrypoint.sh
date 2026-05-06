#!/bin/sh
# All-in-one supervisor: download model if needed, start whisper-server in
# the background on 127.0.0.1:8080, run the OpenAI-compat shim in the
# foreground on 0.0.0.0:8000. Exit if either child dies.
set -eu

: "${WHISPER_MODEL:=large-v3}"
: "${WHISPER_THREADS:=4}"
: "${WHISPER_LANGUAGE:=auto}"

MODEL_FILE="/models/ggml-${WHISPER_MODEL}.bin"
if [ ! -f "$MODEL_FILE" ]; then
  echo "[entrypoint] downloading model: $WHISPER_MODEL"
  download-ggml-model.sh "$WHISPER_MODEL" /models
fi

echo "[entrypoint] starting whisper-server"
whisper-server \
  -m "$MODEL_FILE" \
  --host 127.0.0.1 \
  --port 8080 \
  -t "$WHISPER_THREADS" \
  -l "$WHISPER_LANGUAGE" &
WHISPER_PID=$!

cleanup() {
  kill -TERM "$WHISPER_PID" "${SHIM_PID:-}" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup TERM INT

# Wait until whisper-server is accepting connections before starting the shim.
for _ in $(seq 1 600); do
  if curl -fsS -o /dev/null "http://127.0.0.1:8080/" 2>/dev/null; then
    break
  fi
  if ! kill -0 "$WHISPER_PID" 2>/dev/null; then
    echo "[entrypoint] whisper-server died during startup" >&2
    wait "$WHISPER_PID" || exit $?
    exit 1
  fi
  sleep 1
done

echo "[entrypoint] starting shim"
WHISPER_URL="http://127.0.0.1:8080" \
  WHISPER_MODEL_NAME="${WHISPER_MODEL_NAME:-$WHISPER_MODEL}" \
  exec uvicorn main:app --host 0.0.0.0 --port 8000
