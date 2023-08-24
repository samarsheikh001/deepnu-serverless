import cv2
import numpy as np
from io import BytesIO
import os
import subprocess
import time
from PIL import Image
import requests

from dotenv import load_dotenv

from scripts.storage import download_file_from_bucket
load_dotenv()


CACHE_FOLDER = "cache"
SEG_MODEL = "cloth_segm.pth"
SD_MODEL = "uber_inpainting.safetensors"
UberVag_LORA = "UberVag_LORA_V1.0.safetensors"
UberDatAss_LORA = "UberDatAss_LORA_V1.0.safetensors"
UberRealVag_LORA = "UberRealVag_LORA_V1.0.safetensors"


def ensure_model_exists(model_name):
    model_path = os.path.join(CACHE_FOLDER, model_name)
    if not os.path.exists(model_path):
        download_file_from_bucket(model_name, model_path, bucket_name="models")


def download_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")


def load_image(path):
    return Image.open(path).convert("RGB")


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest])
    print("downloading took: ", time.time() - start)


def dilate_mask(input_image, structure_size, iterations):
    # Convert PIL image to OpenCV image (grayscale)
    image = np.array(input_image.convert('L'))

    # Define the structure for dilation
    structure = np.ones((structure_size, structure_size), np.uint8)

    # Apply dilation
    image_dilated = cv2.dilate(image, structure, iterations=iterations)

    # # Save the result
    # cv2.imwrite("seg_dialated.png", image_dilated)

    # Convert to PIL Image
    image_pil = Image.fromarray(image_dilated)

    return image_pil
