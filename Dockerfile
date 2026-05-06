ARG WHISPER_REF=master
# Default targets Sapphire Rapids ISA — AMX_TILE/INT8/BF16 + AVX-512 + AVX-VNNI.
# Compiler emits these instructions even when the build host lacks them, so
# GH-hosted runners (which don't have AMX) can still produce this image.
# Resulting binary will SIGILL on hosts older than Sapphire Rapids.
ARG TARGET_ARCH=sapphirerapids

FROM debian:bookworm-slim AS whisper-build
ARG WHISPER_REF
ARG TARGET_ARCH
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git ca-certificates curl libcurl4-openssl-dev pkg-config \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /src
RUN git clone --depth 1 --branch "${WHISPER_REF}" https://github.com/ggml-org/whisper.cpp.git .
RUN cmake -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=OFF \
      -DGGML_NATIVE=OFF \
      -DCMAKE_C_FLAGS="-march=${TARGET_ARCH}" \
      -DCMAKE_CXX_FLAGS="-march=${TARGET_ARCH}" \
      -DGGML_AMX=ON \
      -DGGML_AMX_TILE=ON \
      -DGGML_AMX_INT8=ON \
      -DGGML_AMX_BF16=ON \
      -DWHISPER_BUILD_SERVER=ON \
      -DWHISPER_BUILD_TESTS=OFF \
      -DWHISPER_BUILD_EXAMPLES=ON \
 && cmake --build build -j"$(nproc)" --target whisper-server

# Runtime: Python + whisper-server + shim, supervised by tini.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 libcurl4 ca-certificates curl ffmpeg tini \
 && rm -rf /var/lib/apt/lists/*

COPY --from=whisper-build /src/build/bin/whisper-server /usr/local/bin/whisper-server
COPY --from=whisper-build /src/models/download-ggml-model.sh /usr/local/bin/download-ggml-model.sh
RUN chmod +x /usr/local/bin/download-ggml-model.sh

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py .
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

VOLUME ["/models"]
EXPOSE 8000
ENTRYPOINT ["/usr/bin/tini", "--", "/entrypoint.sh"]
