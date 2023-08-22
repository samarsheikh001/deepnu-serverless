FROM pytorch/pytorch:latest

WORKDIR /app

RUN pip install runpod

ADD handler.py .

# Update system and install necessary system packages for git and wget
RUN apt-get update && apt-get install -y git wget && rm -rf /var/lib/apt/lists/*

# Clone the repository
RUN git clone -b run-dreambooth https://github.com/samarsheikh001/job_queue_template .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Run celery worker when the container launches
CMD [ "python", "-u", "/handler.py" ]