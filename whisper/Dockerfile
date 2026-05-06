ARG WHISPER_REF=master

FROM debian:bookworm-slim AS build
ARG WHISPER_REF
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git ca-certificates curl libcurl4-openssl-dev pkg-config \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone --depth 1 --branch "${WHISPER_REF}" https://github.com/ggml-org/whisper.cpp.git .

# GGML_NATIVE=ON => -march=native; picks up AMX/AVX-512/VNNI from the build host.
# This image is therefore tied to the build host's ISA — only ship to CPUs that match.
RUN cmake -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=OFF \
      -DGGML_NATIVE=ON \
      -DGGML_AMX=ON \
      -DWHISPER_BUILD_SERVER=ON \
      -DWHISPER_BUILD_TESTS=OFF \
      -DWHISPER_BUILD_EXAMPLES=ON \
 && cmake --build build -j"$(nproc)" --target whisper-server

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 libcurl4 ca-certificates curl ffmpeg \
 && rm -rf /var/lib/apt/lists/*

COPY --from=build /src/build/bin/whisper-server /usr/local/bin/whisper-server
COPY --from=build /src/models/download-ggml-model.sh /usr/local/bin/download-ggml-model.sh
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh /usr/local/bin/download-ggml-model.sh

VOLUME ["/models"]
EXPOSE 8080
ENTRYPOINT ["/entrypoint.sh"]
