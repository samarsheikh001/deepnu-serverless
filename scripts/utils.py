from scripts.storage import download_file_from_s3
from io import BytesIO
import os
import subprocess
import time
from PIL import Image
import requests

from dotenv import load_dotenv
load_dotenv()


CACHE_FOLDER = "model"
SEG_MODEL = "cloth_segm.pth"
SD_MODEL_CACHE = "uber_inpainting.safetensors"
SD_MODEL_URL = "https://deepnu.sfo3.digitaloceanspaces.com/uberRealisticPornMerge_urpmv13Inpainting.safetensors"


def download_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")


def load_image(path):
    return Image.open(path).convert("RGB")


def setup():
    os.makedirs(CACHE_FOLDER, exist_ok=True)
    if not os.path.exists(f"{CACHE_FOLDER}/{SEG_MODEL}"):
        download_file_from_s3(SEG_MODEL, f"{CACHE_FOLDER}/{SEG_MODEL}")
    if not os.path.exists(f"{CACHE_FOLDER}/{SD_MODEL_CACHE}"):
        download_file_from_s3(
            SD_MODEL_CACHE, f"{CACHE_FOLDER}/{SD_MODEL_CACHE}")


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest])
    print("downloading took: ", time.time() - start)
