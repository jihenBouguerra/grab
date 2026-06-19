# Stage 1 — build whisper.cpp for Linux x86_64
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y \
    git cmake make g++ curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
RUN git clone --depth 1 https://github.com/ggerganov/whisper.cpp .

# Build without GPU (Cloud Run has no GPU)
RUN cmake -B build -DGGML_NATIVE=OFF -DGGML_METAL=OFF -DGGML_CUDA=OFF && \
    cmake --build build --config Release -j$(nproc) --target whisper-cli

# Download the base model (~142 MB)
RUN bash models/download-ggml-model.sh base


# Stage 2 — slim runtime image
FROM python:3.11-slim

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Whisper binary + model from builder
COPY --from=builder /build/build/bin/whisper-cli bin/whisper-cli
COPY --from=builder /build/models/ggml-base.bin  models/ggml-base.bin
RUN chmod +x bin/whisper-cli

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code and frontend
COPY app.py transcript.py ./
COPY static/ static/

# Cloud Run injects PORT; fall back to 8080
ENV PORT=8080
EXPOSE 8080

CMD python app.py
