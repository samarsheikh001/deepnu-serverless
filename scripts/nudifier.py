from io import BytesIO
import requests
from PIL import Image

def download_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")