FROM nvidia/cuda:11.7.1-devel-ubuntu20.04

ARG USE_CUDA=1
ARG TORCH_ARCH="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    AM_I_DOCKER=True \
    BUILD_WITH_CUDA="${USE_CUDA}" \
    TORCH_CUDA_ARCH_LIST="${TORCH_ARCH}" \
    CUDA_HOME=/usr/local/cuda-11.7/

WORKDIR /app

RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y python3 python3-pip git wget python3-dev build-essential libgl1-mesa-glx libglib2.0-0 ffmpeg nano cmake libpython3-dev && \
    rm -rf /var/lib/apt/lists/* && \
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

RUN git clone https://github.com/IDEA-Research/Grounded-Segment-Anything && \
    cd Grounded-Segment-Anything && \
    pip3 install --no-cache-dir -e segment_anything -e GroundingDINO && \
    cd ../ && \
    rm -rf Grounded-Segment-Anything

COPY . /app

RUN pip3 install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

