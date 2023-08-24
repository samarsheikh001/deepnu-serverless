FROM pytorch/pytorch:latest

WORKDIR /app

# Update system and install necessary system packages for git and wget
RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y git wget python3-dev build-essential libgl1-mesa-glx
RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Clone the repository
RUN git clone https://github.com/samarsheikh001/deepnu-serverless .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Run celery worker when the container launches
CMD [ "python", "-u", "handler.py" ]