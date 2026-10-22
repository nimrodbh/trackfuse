FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends g++ make && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . .
RUN make -C core build/libtrackfuse.so && pip install --no-cache-dir -e ".[dev]"
CMD ["python", "-m", "trackfuse", "demo"]
