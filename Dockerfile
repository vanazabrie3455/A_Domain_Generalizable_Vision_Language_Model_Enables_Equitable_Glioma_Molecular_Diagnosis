FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime
RUN apt-get update && apt-get install -y --no-install-recommends libopenslide0 && apt-get clean && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY requirements.txt pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .
COPY configs ./configs
COPY scripts ./scripts
ENTRYPOINT ["dgvlm-train"]

