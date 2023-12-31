FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/London

WORKDIR /app

# Update system and install necessary system packages for git and wget
RUN apt-get update && apt-get install -y git wget python3-dev build-essential libgl1-mesa-glx libglib2.0-0 ffmpeg nano
RUN apt-get install -y cmake libpython3-dev
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Clone the repository
RUN git clone https://github.com/samarsheikh001/deepnu-serverless .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install git+https://github.com/IDEA-Research/GroundingDINO.git

# Run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]