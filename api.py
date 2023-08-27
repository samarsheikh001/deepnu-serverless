from io import BytesIO
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict

from typing import Optional
from scripts.cloth_seg import get_clothes_mask
from scripts.predictor import Predictor
from scripts.storage import upload_in_memory_object
from scripts.utils import download_image

from dotenv import load_dotenv
load_dotenv()

app = FastAPI()


class ImageUrl(BaseModel):
    image_url: str
    prompt: Optional[str] = "RAW photo of a nude woman, naked, <lora:UberRealVag_LORA_V1.0:0.7> <lora:UberDatAss_LORA_V1.0:0.7> <lora:UberVag_LORA_V1.0:0.5>"
    negative_prompt: Optional[str] = None
    scale_down_value: Optional[int] = 512
    steps: Optional[int] = 25
    seed: Optional[int] = None
    dilate_value: Optional[int] = 5
    remove_mask_prompt: Optional[str] = 'cloth . dress . bra . panty'
    should_upscale: Optional[bool] = True


pred = Predictor()


@app.post("/process_image")
async def process_image(image_data: ImageUrl) -> Dict[str, str]:
    try:
        img_url = image_data.image_url
        prompt = image_data.prompt
        negative_prompt = image_data.negative_prompt
        scale_down_value = image_data.scale_down_value
        steps = image_data.steps
        seed = image_data.seed
        dilate_value = image_data.dilate_value
        remove_mask_prompt = image_data.remove_mask_prompt
        should_upscale = image_data.should_upscale

        image = download_image(img_url)

        out_img = pred.predict(image, prompt,
                               negative_prompt, remove_mask_prompt, scale_down_value, steps, seed, dilate_value, should_upscale)
        # Convert your image into bytes
        byte_arr = BytesIO()
        out_img.save(byte_arr, format='PNG')
        img_bytes = byte_arr.getvalue()
        # Generate a unique filename
        image_name = f"{uuid.uuid4().hex}.png"
        image_url = upload_in_memory_object(image_name, img_bytes)
        return {"result": image_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
