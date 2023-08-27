from RealESRGAN import RealESRGAN
from PIL import Image
import torch


class ImageUpscaler:
    def __init__(self, device='cuda', scale=2, model_path='cache/upscaler/RealESRGAN_x2.pth'):
        self.device = torch.device(device)
        self.model = RealESRGAN(self.device, scale=scale)
        self.model.load_weights(model_path, download=True)

    def upscale(self, image):
        return self.model.predict(image)
