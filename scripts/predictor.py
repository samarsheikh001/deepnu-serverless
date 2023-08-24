from scripts.utils import CACHE_FOLDER, SD_MODEL, SEG_MODEL, UberDatAss_LORA, UberRealVag_LORA, UberVag_LORA, dilate_mask, ensure_model_exists
from scripts.cloth_seg import get_clothes_mask
from diffusers import StableDiffusionInpaintPipeline
import torch
import math
import os
from dotenv import load_dotenv
# from scripts.storage import upload_image
load_dotenv()


class Predictor():
    def __init__(self):
        self.setup()

    def setup(self):
        os.makedirs(CACHE_FOLDER, exist_ok=True)
        # Ensure SEG_MODEL and SD_MODEL_CACHE exist
        ensure_model_exists(SEG_MODEL)
        ensure_model_exists(SD_MODEL)
        ensure_model_exists(UberVag_LORA)
        ensure_model_exists(UberDatAss_LORA)
        ensure_model_exists(UberRealVag_LORA)

        pipe = StableDiffusionInpaintPipeline.from_single_file(
            f"{CACHE_FOLDER}/{SD_MODEL}",
            revision="fp16",
            torch_dtype=torch.float16,
        )

        pipe.safety_checker = None
        # pipe.enable_attention_slicing()
        self.pipe = pipe.to("cuda")

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

    def predict(self, image, prompt, negative_prompt, scale_down_value=768, steps=25, seed=None):
        if (seed == 0) or (seed == None):
            seed = int.from_bytes(os.urandom(2), byteorder='big')
        generator = torch.Generator('cuda').manual_seed(seed)
        print("Using seed:", seed)
        r_image = self.scale_down_image(image, scale_down_value)
        r_mask = dilate_mask(get_clothes_mask(r_image), 30, 1)
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
