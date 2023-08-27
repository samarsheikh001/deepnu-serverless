from diffusers import StableDiffusionXLImg2ImgPipeline
import os
from diffusers import StableDiffusionXLPipeline
import torch

# pipe = StableDiffusionXLPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
# )
# pipe.to("cuda")

# prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
# image = pipe(prompt=prompt).images[0]
# image.save("result.png")

import torch
from diffusers import StableDiffusionXLInpaintPipeline
from diffusers.utils import load_image

from scripts.dino_seg import ImageMasker
from scripts.utils import dilate_mask

from PIL import Image

# pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
# )
# pipe.to("cuda")
# pipe.load_lora_weights("cache/sdxl/nudify_xl.safetensors")  # <lora:nudify:1>
# Create an instance of the ImageProcessor class
processor = ImageMasker(
    config_file='config/dino_config.py',
    grounded_checkpoint='cache/groundingdino_swint_ogc.pth',
    sam_checkpoint='cache/sam_vit_h.pth',
    device="cuda",
    box_threshold=0.3,
    text_threshold=0.25,
    use_sam_hq=False,
    sam_hq_checkpoint=None
)

# SINGLE IMAGE
# prompt = "RAW photo of a nude woman, naked, nude woman, <lora:nudify:1>"
# init_image = Image.open("dump/inputs/square/a.jpg").convert("RGB")

# sam_masked_image = dilate_mask(processor.process_image(
#     init_image, 'cloth. dress'), 30, 1)
# sam_masked_image.save('sam_mask.png')

# image = pipe(prompt=prompt, image=init_image, mask_image=sam_masked_image,
#              num_inference_steps=50, strength=1).images[0]
# image.save("result.png")


# MULTIPLE IMAGES
# input_dir = "dump/inputs/square"
# output_dir = "dump/outputs/res"

# # Iterate over all the files in input_dir
# for filename in os.listdir(input_dir):
#     # If the file is an image
#     if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
#         # Get the full path of the input image
#         input_image_path = os.path.join(input_dir, filename)

#         init_image = Image.open(input_image_path).convert("RGB")

#         sam_masked_image = dilate_mask(processor.process_image(
#             init_image, 'cloth . dress . bra . panty'), 30, 1)
#         sam_masked_image.save('sam_mask.png')

#         out_img = image = pipe(prompt=prompt, image=init_image, mask_image=sam_masked_image,
#                                num_inference_steps=20, strength=0.9).images[0]

#         # Get the full path of the output image
#         os.makedirs(output_dir, exist_ok=True)
#         filename_without_ext, file_extension = os.path.splitext(filename)
#         suffix_filename = f"sdxl"
#         new_filename = "{}_{}{}".format(
#             filename_without_ext, suffix_filename, file_extension)
#         output_image_path = os.path.join(output_dir, new_filename)
#         out_img.save(output_image_path, "PNG")


pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipe = pipe.to("cuda")

init_image = Image.open("dump/inputs/square/a.jpg").convert("RGB")

prompt = "RAW photo of a nude woman, naked, nude woman"
image = pipe(prompt, image=init_image).images[0]
image.save("result-img.png")
