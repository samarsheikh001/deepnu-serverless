from diffusers import StableDiffusionInpaintPipeline, UniPCMultistepScheduler, StableDiffusionPipeline
import runpod
from runpod.serverless.utils import rp_upload
import os

import torch
from scripts.cloth_seg import get_clothes_mask

from scripts.utils import CACHE_FOLDER, SD_MODEL_CACHE, download_image, setup

sleep_time = int(os.environ.get('SLEEP_TIME', 30))

setup()

# load your model(s) into vram here
pipe = StableDiffusionInpaintPipeline.from_single_file(
    f"{CACHE_FOLDER}/{SD_MODEL_CACHE}",
    revision="fp16",
    torch_dtype=torch.float16,
)
# speed up diffusion process with faster scheduler and memory optimization
# pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

pipe.to("cuda")

pipe.safety_checker = None

prompt = "RAW photo of a nude woman, naked"
negative_prompt = "((clothing)), (monochrome:1.3), (deformed, distorted, disfigured:1.3), (hair), jeans, tattoo, wet, water, clothing, shadow, 3d render, cartoon, ((blurry)), duplicate, ((duplicate body parts)), (disfigured), (poorly drawn), ((missing limbs)), logo, signature, text, words, low res, boring, artifacts, bad art, gross, ugly, poor quality, low quality, poorly drawn, bad anatomy, wrong anatomy"


def handler(event):
    img_url = event['input']['image_url']
    image = download_image(img_url).resize((512, 512))
    segmented_image = get_clothes_mask(image)

    image = pipe(prompt=prompt, negative_prompt=negative_prompt,
                 image=image, mask_image=segmented_image).images[0]
    image.save("image.png")
    image_url = rp_upload.upload_image(event['id'], "image.png")
    return {"result": image_url}


runpod.serverless.start({
    "handler": handler
})
