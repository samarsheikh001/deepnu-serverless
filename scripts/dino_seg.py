import argparse
import io
import os
import copy

import numpy as np
import json
import torch
from PIL import Image, ImageDraw, ImageFont

# Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

from segment_anything import (
    build_sam,
    SamPredictor
)
import numpy as np
import matplotlib.pyplot as plt


class ImageMasker:
    def __init__(self, config_file, grounded_checkpoint, sam_checkpoint, box_threshold=0.3, text_threshold=0.25, device="cpu", use_sam_hq=False, sam_hq_checkpoint=None):
        self.config_file = config_file
        self.grounded_checkpoint = grounded_checkpoint
        self.sam_checkpoint = sam_checkpoint
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device
        self.output_dir = "outputs"
        self.use_sam_hq = use_sam_hq
        self.sam_hq_checkpoint = sam_hq_checkpoint

        self.model = self.load_model(
            self.config_file, self.grounded_checkpoint, device=self.device)

        if self.use_sam_hq:
            pass
            # self.predictor = SamPredictor(build_sam_hq(
            #     checkpoint=self.sam_hq_checkpoint).to(self.device))
        else:
            self.predictor = SamPredictor(
                build_sam(checkpoint=self.sam_checkpoint).to(self.device))

    def load_image(self, image_path):
        image_pil = Image.open(image_path).convert("RGB")
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(image_pil, None)
        return image_pil, image

    def load_model(self, model_config_path, model_checkpoint_path, device):
        args = SLConfig.fromfile(model_config_path)
        args.device = device
        model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
        load_res = model.load_state_dict(
            clean_state_dict(checkpoint["model"]), strict=False)
        print(load_res)
        _ = model.eval()
        return model

    def get_grounding_output(self, image, caption, with_logits=True):
        caption = caption.lower().strip()
        if not caption.endswith("."):
            caption = caption + "."
        self.model = self.model.to(self.device)
        image = image.to(self.device)
        with torch.no_grad():
            outputs = self.model(image[None], captions=[caption])
        logits = outputs["pred_logits"].cpu().sigmoid()[0]
        boxes = outputs["pred_boxes"].cpu()[0]
        logits.shape[0]

        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > self.box_threshold
        logits_filt = logits_filt[filt_mask]
        boxes_filt = boxes_filt[filt_mask]
        logits_filt.shape[0]

        tokenlizer = self.model.tokenizer
        tokenized = tokenlizer(caption)

        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(
                logit > self.text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(
                    pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)

        return boxes_filt, pred_phrases

    def process_image(self, image_pil, text_prompt):
        # os.makedirs(self.output_dir, exist_ok=True)
        # image_pil.save(os.path.join(self.output_dir, "raw_image.jpg"))
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(image_pil, None)
        boxes_filt, pred_phrases = self.get_grounding_output(
            image, text_prompt)

        image = np.array(image_pil)
        if image.shape[2] == 4:
            image = image[..., :3]
        self.predictor.set_image(image)

        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        transformed_boxes = self.predictor.transform.apply_boxes_torch(
            boxes_filt, image.shape[:2]).to(self.device)

        masks, _, _ = self.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(self.device),
            multimask_output=False,
        )

        canvas = np.zeros_like(image[:, :, 0])
        for mask in masks:
            mask_image = mask.cpu().numpy()[0]
            canvas[mask_image == 1] = 255

        img = Image.fromarray(canvas.astype(np.uint8))
        return img

# def process_image(self, image_path, text_prompt):
    #     os.makedirs(self.output_dir, exist_ok=True)
    #     image_pil, image = self.load_image(image_path)
    #     image_pil.save(os.path.join(self.output_dir, "raw_image.jpg"))
    #     boxes_filt, pred_phrases = self.get_grounding_output(
    #         image, text_prompt)

    #     image = cv2.imread(image_path)
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     self.predictor.set_image(image)

    #     size = image_pil.size
    #     H, W = size[1], size[0]
    #     for i in range(boxes_filt.size(0)):
    #         boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
    #         boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
    #         boxes_filt[i][2:] += boxes_filt[i][:2]

    #     boxes_filt = boxes_filt.cpu()
    #     transformed_boxes = self.predictor.transform.apply_boxes_torch(
    #         boxes_filt, image.shape[:2]).to(self.device)

    #     masks, _, _ = self.predictor.predict_torch(
    #         point_coords=None,
    #         point_labels=None,
    #         boxes=transformed_boxes.to(self.device),
    #         multimask_output=False,
    #     )

    #     canvas = np.zeros_like(image[:, :, 0])
    #     for mask in masks:
    #         mask_image = mask.cpu().numpy()[0]
    #         canvas[mask_image == 1] = 255

    #     img = Image.fromarray(canvas.astype(np.uint8))
    #     return img
