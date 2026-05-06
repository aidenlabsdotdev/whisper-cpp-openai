#!/bin/sh
set -eu
: "${WHISPER_MODEL:=large-v3}"
: "${WHISPER_THREADS:=4}"
: "${WHISPER_LANGUAGE:=auto}"
MODEL_FILE="/models/ggml-${WHISPER_MODEL}.bin"
if [ ! -f "$MODEL_FILE" ]; then
  echo "Downloading model: $WHISPER_MODEL"
  download-ggml-model.sh "$WHISPER_MODEL" /models
fi
exec whisper-server \
  -m "$MODEL_FILE" \
  --host 0.0.0.0 \
  --port 8080 \
  -t "$WHISPER_THREADS" \
  -l "$WHISPER_LANGUAGE"
