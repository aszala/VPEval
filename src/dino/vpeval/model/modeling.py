####
# This source code is adapted from the official GroundingDINO [repo](https://github.com/IDEA-Research/GroundingDINO).

import os
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
import torch.nn as nn
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from transformers import AutoImageProcessor, DPTForDepthEstimation

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.args = args

        config_path = args.grounding_dino_config_path
        weights_path = args.grounding_dino_weights_path

        self.model = self.load_model(config_path, weights_path, device=args.device)
        
        dpt_checkpoint: str = "Intel/dpt-large"

        self.dpt_image_processor = AutoImageProcessor.from_pretrained(dpt_checkpoint)
        self.depth_model = DPTForDepthEstimation.from_pretrained(dpt_checkpoint)

    def load_model(self, model_config_path, model_checkpoint_path, device="cpu"):
        args = SLConfig.fromfile(model_config_path)
        args.device = device
        model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
        load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        print(load_res)
        _ = model.eval()
        return model

    def load_image(self, image_path):
        # load image
        image_pil = Image.open(image_path).convert("RGB")  # load image

        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(image_pil, None)  # 3, h, w
        return image_pil, image

    def get_grounding_output(self, model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        model = model.to(device)
        image = image.to(device)
        with torch.no_grad():
            outputs = model(image[None], captions=[caption])
        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
        logits.shape[0]

        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        logits_filt.shape[0]

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)

        return boxes_filt, pred_phrases

    def plot_boxes_to_image(self, image_pil, tgt):
        H, W = tgt["size"]
        boxes = tgt["boxes"]
        labels = tgt["labels"]
        assert len(boxes) == len(labels), "boxes and labels must have same length"

        draw = ImageDraw.Draw(image_pil)
        mask = Image.new("L", image_pil.size, 0)
        mask_draw = ImageDraw.Draw(mask)

        # draw boxes and masks
        for box, label in zip(boxes, labels):
            # from 0..1 to 0..W, 0..H
            box = box * torch.Tensor([W, H, W, H])
            # from xywh to xyxy
            box[:2] -= box[2:] / 2
            box[2:] += box[:2]
            # random color
            color = tuple(np.random.randint(0, 255, size=3).tolist())
            # draw
            x0, y0, x1, y1 = box
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

            draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
            # draw.text((x0, y0), str(label), fill=color)

            font = ImageFont.load_default()
            if hasattr(font, "getbbox"):
                bbox = draw.textbbox((x0, y0), str(label.split("(")[0]), font)
            else:
                w, h = draw.textsize(str(label.split("(")[0]), font)
                bbox = (x0, y0, w + x0, y0 + h)
            # bbox = draw.textbbox((x0, y0), str(label))
            draw.rectangle(bbox, fill=color)
            draw.text((x0, y0), str(label.split("(")[0]), fill="white")

            mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

        return image_pil, mask

    def get_object_depth(self, detection_labels, detection_boxes, depth_maps):
        B: int = len(detection_labels)
        
        for b in range(B):
            depth_ranks = []
            image = np.array(depth_maps[b]).copy()

            for i in range(len(detection_labels[b])):
                x, y, x2, y2 = detection_boxes[b][i]

                image_new = image[y:y2,x:x2]
                depth = np.average(image_new)
                depth_ranks.append(depth)
                
            depth_ranks = np.argsort(depth_ranks)

            for i in range(len(detection_boxes[b])):
                detection_boxes[b][i].append(depth_ranks[i])

        return detection_labels, detection_boxes
    
    def extract_depth(self, image):
        inputs = self.dpt_image_processor(images=image, return_tensors="pt")
        inputs = inputs.to(self.args.device)

        with torch.no_grad():
            outputs = self.depth_model(**inputs)
            predicted_depth = outputs.predicted_depth

        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        )
        output = prediction.squeeze().cpu().numpy()
        formatted = (output * 255 / np.max(output)).astype("uint8")
        depth = Image.fromarray(formatted)

        return depth

    def forward(self, datum, box_threshold=0.40, text_threshold=0.25):
        B = len(datum)
        
        images = []
        boxes = []
        labels = []
        sizes = []
        depths = []

        for b in range(B):
            image_source, image = self.load_image(datum[b]["image_path"])
            
            text_prompt = ','.join(datum[b]["gt_labels"])
            text_prompts = text_prompt.split(",")

            boxes_filt = []
            pred_phrases = []
            for text_prompt in text_prompts:
                b_filt, p_phrases = self.get_grounding_output(
                    self.model, image, text_prompt, box_threshold, text_threshold, device=self.args.device
                )

                for i, box in enumerate(list(b_filt.numpy())):
                    boxes_filt.append(list(box))
                    pred_phrases.append(p_phrases[i])
                    
            boxes_filt = torch.Tensor(boxes_filt)
            
            size = image_source.size
            pred_dict = {
                "boxes": boxes_filt,
                "size": [size[1], size[0]],  # H,W
                "labels": pred_phrases,
            }

            image_with_box = self.plot_boxes_to_image(image_source, pred_dict)[0]
            images.append(image_with_box)

            H, W = (size[1], size[0])

            for i in range(len(boxes_filt)):
                box = boxes_filt[i] * torch.Tensor([W, H, W, H])
                # from xywh to xyxy
                box[:2] -= box[2:] / 2
                box[2:] += box[:2]

                box = [ int(x) for x in box ]

                boxes_filt[i] = torch.Tensor(box)

            boxes_filt = [ list(x) for x in boxes_filt ]
            for box in boxes_filt:
                for i, b in enumerate(box):
                    box[i] = int(b)

            for i, p in enumerate(pred_phrases):
                x = p.split("(")[0]
                pred_phrases[i] = x

            boxes.append(boxes_filt)
            labels.append(pred_phrases)
            sizes.append([size[1], size[0]])

            depth = self.extract_depth(image_source)

            depths.append(depth)
        
        final_labels, final_boxes = self.get_object_depth(labels, boxes, depths)

        return final_labels, final_boxes, images
