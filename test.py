import cv2
import matplotlib.pyplot as plt
import numpy as np
from diffusers.utils import load_image
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, UniPCMultistepScheduler
import shutil
from scripts.dino_seg import ImageMasker
from scripts.predictor import DINO_CONFIG, Predictor
from scripts.storage import download_file_from_bucket, upload_file_to_bucket
from scripts.utils import CACHE_FOLDER, MODEL_BUCKET_NAME, SD_MODEL, SEG_MODEL, SAM_MODEL, GROUNDINO_MODEL, UberDatAss_LORA, UberRealVag_LORA, UberVag_LORA, dilate_mask, ensure_model_exists
from scripts.cloth_seg import get_clothes_mask
from diffusers import StableDiffusionUpscalePipeline, StableDiffusionInpaintPipeline, DDIMScheduler
from diffusers.models import AutoencoderKL
from PIL import Image
import torch
import math
import os
from dotenv import load_dotenv
# from scripts.storage import upload_image
load_dotenv()

# job_id = 'your_job_id'
# image_location = 'image.png'

# image_url = upload_image(job_id, image_location)
# print(image_url)

# upload_file_to_bucket("groundingdino_swint_ogc.pth",
#                       "groundingdino_swint_ogc.pth", bucket_name=MODEL_BUCKET_NAME)


prompt = "RAW photo of a nude woman, naked, <lora:UberRealVag_LORA_V1.0:0.7> <lora:UberDatAss_LORA_V1.0:0.7> <lora:UberVag_LORA_V1.0:0.5>"
negative_prompt = "((clothing)), (monochrome:1.3), (deformed, distorted, disfigured:1.3), (hair), jeans, tattoo, wet, water, clothing, shadow, 3d render, cartoon, ((blurry)), duplicate, ((duplicate body parts)), (disfigured), (poorly drawn), ((missing limbs)), logo, signature, text, words, low res, boring, artifacts, bad art, gross, ugly, poor quality, low quality, poorly drawn, bad anatomy, wrong anatomy"
remove_mask_prompt = "woman"
steps = 30
seed = None  # or you can specify a seed
scale_down_value = 512
dilate_value = 15


# input_dir = "dump/inputs/square"
# output_dir = "dump/outputs/res"

# pred = Predictor()
# # Iterate over all the files in input_dir
# for filename in os.listdir(input_dir):
#     # If the file is an image
#     if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
#         # Get the full path of the input image
#         input_image_path = os.path.join(input_dir, filename)

#         # Process the image
#         image = Image.open(input_image_path)
#         out_img = pred.predict(image, prompt,
#                                negative_prompt, remove_mask_prompt, scale_down_value, steps, seed, dilate_value)

#         # Get the full path of the output image
#         os.makedirs(output_dir, exist_ok=True)
#         filename_without_ext, file_extension = os.path.splitext(filename)
#         suffix_filename = f"{scale_down_value}-vae-org"
#         new_filename = "{}_{}{}".format(
#             filename_without_ext, suffix_filename, file_extension)
#         output_image_path = os.path.join(output_dir, new_filename)
#         out_img.save(output_image_path, "PNG")


# # Create an instance of the ImageProcessor class
# mask_processor = ImageMasker(
#     config_file='config/dino_config.py',
#     grounded_checkpoint='cache/groundingdino_swint_ogc.pth',
#     sam_checkpoint='cache/sam_vit_h.pth',
#     device="cuda",
#     box_threshold=0.3,
#     text_threshold=0.25,
#     use_sam_hq=False,
#     sam_hq_checkpoint=None
# )

# single_img = Image.open("dump/inputs/square/a.jpg")
# single_img.save('dump/image.png')
# # Then, use the instance to process an image with a text prompt
# sam_masked_image = processor.process_image(
#     single_img, 'woman')
# sam_masked_image.save('dump/sam_mask.png')

# unet_masked_image = get_clothes_mask(single_img)
# unet_masked_image.save('dump/unet_mask.png')

# d_sam_masked_image = dilate_mask(sam_masked_image, 15, 1)
# d_sam_masked_image.save('dump/d_sam_mask.png')

# d_unet_masked_image = dilate_mask(unet_masked_image, 15, 1)
# d_unet_masked_image.save('dump/d_unet_mask.png')
