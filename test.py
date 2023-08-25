import shutil
from scripts.dino_seg import ImageMasker
from scripts.predictor import DINO_CONFIG
from scripts.storage import download_file_from_bucket, upload_file_to_bucket
from scripts.utils import CACHE_FOLDER, MODEL_BUCKET_NAME, SD_MODEL, SEG_MODEL, SAM_MODEL, GROUNDINO_MODEL, UberDatAss_LORA, UberRealVag_LORA, UberVag_LORA, dilate_mask, ensure_model_exists
from scripts.cloth_seg import get_clothes_mask
from diffusers import StableDiffusionUpscalePipeline, StableDiffusionInpaintPipeline
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


class Predictor():
    def __init__(self):
        self.setup()

    def setup(self):
        os.makedirs(CACHE_FOLDER, exist_ok=True)
        # Ensure models exist
        ensure_model_exists(SEG_MODEL)
        ensure_model_exists(SD_MODEL)
        ensure_model_exists(UberVag_LORA)
        ensure_model_exists(UberDatAss_LORA)
        ensure_model_exists(UberRealVag_LORA)
        ensure_model_exists(SAM_MODEL)
        ensure_model_exists(GROUNDINO_MODEL)

        pipe = StableDiffusionInpaintPipeline.from_single_file(
            f"{CACHE_FOLDER}/{SD_MODEL}",
            revision="fp16",
            torch_dtype=torch.float16,
        )

        pipe.safety_checker = None
        # pipe.enable_attention_slicing()
        self.pipe = pipe.to("cuda")

        self.processor = ImageMasker(
            config_file=DINO_CONFIG,
            grounded_checkpoint=f'{CACHE_FOLDER}/{GROUNDINO_MODEL}',
            sam_checkpoint=f'{CACHE_FOLDER}/{SAM_MODEL}',
            device="cuda",
            box_threshold=0.3,
            text_threshold=0.25,
            use_sam_hq=False,
            sam_hq_checkpoint=None
        )

    def scale_down_image(self, image, max_size):
        width, height = image.size
        scaling_factor = min(max_size/width, max_size/height)
        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)
        resized_image = image.resize((new_width, new_height))
        cropped_image = self.crop_center(resized_image)
        return cropped_image

    def crop_center(self, pil_img):
        img_width, img_height = pil_img.size
        crop_width = self.base(img_width)
        crop_height = self.base(img_height)
        return pil_img.crop(
            (
                (img_width - crop_width) // 2,
                (img_height - crop_height) // 2,
                (img_width + crop_width) // 2,
                (img_height + crop_height) // 2)
        )

    def base(self, x):
        return int(8 * math.floor(int(x)/8))

    def predict(self, image, prompt, negative_prompt, scale_down_value=768, steps=25, seed=None, dilate_value=5):
        if (seed == 0) or (seed == None):
            seed = int.from_bytes(os.urandom(2), byteorder='big')
        generator = torch.Generator('cuda').manual_seed(seed)
        print("Using seed:", seed)
        r_image = self.scale_down_image(image, scale_down_value)
        # r_mask = dilate_mask(get_clothes_mask(r_image), dilate_value, 1)
        r_mask = dilate_mask(self.processor.process_image(
            r_image, 'cloth . dress . bra . panty'), dilate_value, 1)
        width, height = r_image.size
        image = self.pipe(
            prompt=prompt,
            image=r_image,
            mask_image=r_mask,
            width=width,
            height=height,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            generator=generator,
        ).images[0]

        return image


# prompt = "RAW photo of a nude woman, naked, <lora:UberRealVag_LORA_V1.0:0.7> <lora:UberDatAss_LORA_V1.0:0.7> <lora:UberVag_LORA_V1.0:0.5>"
# negative_prompt = "((clothing)), (monochrome:1.3), (deformed, distorted, disfigured:1.3), (hair), jeans, tattoo, wet, water, clothing, shadow, 3d render, cartoon, ((blurry)), duplicate, ((duplicate body parts)), (disfigured), (poorly drawn), ((missing limbs)), logo, signature, text, words, low res, boring, artifacts, bad art, gross, ugly, poor quality, low quality, poorly drawn, bad anatomy, wrong anatomy"
# steps = 20
# seed = None  # or you can specify a seed
# scale_down_value = 512
# dilate_value = 15


# input_dir = "dump/inputs"
# output_dir = "dump/outputs/00"

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
#                                negative_prompt, scale_down_value, steps, seed, dilate_value)

#         # Get the full path of the output image
#         os.makedirs(output_dir, exist_ok=True)
#         filename_without_ext, file_extension = os.path.splitext(filename)
#         new_filename = "{}_sam_dilated{}".format(
#             filename_without_ext, file_extension)
#         output_image_path = os.path.join(output_dir, new_filename)
#         out_img.save(output_image_path, "PNG")

# # Create an instance of the ImageProcessor class
# processor = ImageMasker(
#     config_file='config/dino_config.py',
#     grounded_checkpoint='cache/groundingdino_swint_ogc.pth',
#     sam_checkpoint='cache/sam_vit_h.pth',
#     device="cuda",
#     box_threshold=0.3,
#     text_threshold=0.25,
#     use_sam_hq=False,
#     sam_hq_checkpoint=None
# )

# single_img = Image.open("dump/inputs/r.jpg")
# single_img.save('dump/image.png')
# # Then, use the instance to process an image with a text prompt
# sam_masked_image = processor.process_image(
#     single_img, 'cloth . dress . inner . bra')
# sam_masked_image.save('dump/sam_mask.png')

# unet_masked_image = get_clothes_mask(single_img)
# unet_masked_image.save('dump/unet_mask.png')

# d_sam_masked_image = dilate_mask(sam_masked_image, 15, 1)
# d_sam_masked_image.save('dump/d_sam_mask.png')

# d_unet_masked_image = dilate_mask(unet_masked_image, 15, 1)
# d_unet_masked_image.save('dump/d_unet_mask.png')
