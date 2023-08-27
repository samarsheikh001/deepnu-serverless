from scripts.dino_seg import ImageMasker
from scripts.upscaler import ImageUpscaler
from scripts.utils import CACHE_FOLDER, GROUNDINO_MODEL, SAM_MODEL, SD_MODEL, SEG_MODEL, UberDatAss_LORA, UberRealVag_LORA, UberVag_LORA, dilate_mask, ensure_model_exists
from scripts.cloth_seg import generate_mask, get_clothes_mask, get_palette, load_seg_model
from diffusers import StableDiffusionInpaintPipeline
from diffusers.models import AutoencoderKL
import torch
import math
import os
from dotenv import load_dotenv
# from scripts.storage import upload_image
load_dotenv()

DINO_CONFIG = 'config/dino_config.py'


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

        # url = "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors"
        # vae = AutoencoderKL.from_single_file(
        #     url,
        # )

        pipe = StableDiffusionInpaintPipeline.from_single_file(
            f"{CACHE_FOLDER}/{SD_MODEL}",
            revision="fp16",
            torch_dtype=torch.float16,
            # vae=vae,
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

        self.upscaler = ImageUpscaler(
            scale=2, model_path="cache/RealESRGAN_x2.pth")

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

    def predict(self, image, prompt, negative_prompt, remove_mask_prompt, scale_down_value=768, steps=25, seed=None, dilate_value=5, should_upscale=True):
        if (seed == 0) or (seed == None):
            seed = int.from_bytes(os.urandom(2), byteorder='big')
        generator = torch.Generator('cuda').manual_seed(seed)
        print("Using seed:", seed)
        r_image = self.scale_down_image(image, scale_down_value)
        # r_mask = dilate_mask(get_clothes_mask(r_image), dilate_value, 1)
        r_mask = dilate_mask(self.processor.process_image(
            r_image, remove_mask_prompt), dilate_value, 1)
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
        if should_upscale:
            image = self.upscaler.upscale(image)
        return image
