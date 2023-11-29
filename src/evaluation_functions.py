import os
import sys
sys.path.append(os.getcwd())

import easyocr
from tqdm import tqdm

from dino.vpeval.model.modeling import Model as DinoModel
from types import FunctionType
from lavis.models import load_model_and_preprocess
from PIL import Image, ImageDraw
import torch
import numpy as np

class EvaluationModules:
    def __init__(self, args):
        self.args = args
        self.object_pipeline = DinoModel(args)
        self.object_pipeline = self.object_pipeline.to(args.device)
        self.object_pipeline.eval()

        self.ocr_pipeline = easyocr.Reader([args.ocr_langauge])

        self.count_map = {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4
        }

        if args.enable_vqa_module:
            self.model, self.vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device=args.device)

        self.visual_explainations = { }

        if not os.path.exists(f"{args.visualization_savepath}/images/"):
            os.makedirs(f"{args.visualization_savepath}/images/")

    def get_evaluation_functions():
        return [x for x, y in EvaluationModules.__dict__.items() if type(y) == FunctionType and not x.startswith('_')]

    def get_image_id(self, image):
        return image.split("/")[-1].replace(".png", "").replace(".jpg", "")

    def vqa(self, image, question, answer_choices, target_answer):
        assert len(answer_choices.split(",")) > 1, answer_choices

        question = f"Question: {question} Choices: {answer_choices.replace(',', ', ')} Answer:"

        image_id = self.get_image_id(image)

        image = self.vis_processors["eval"](Image.open(image).convert("RGB")).unsqueeze(0).to(self.args.device)
        answer = self.model.generate({"image": image, "prompt": question})
        answer = [ a.lower() for a in answer ][0]

        correct = answer == target_answer.lower()

        self.visual_explainations[image_id] = { }

        if correct:
            self.visual_explainations[image_id]["text_explaination"] = f"Q: {question} A: {target_answer}"
        else:
            self.visual_explainations[image_id]["text_explaination"] = f"Q: {question} A: not {target_answer}"

        self.visual_explainations[image_id]["visual_explaination"] = f"n/a"

        return correct

    def objDet(self, image: str, target_objects: str, **kwargs):
        datum = {
            "image_id": self.get_image_id(image),
            "image_path": image,
            "gt_labels": target_objects.split(",")
        }

        predicted_labels, predicted_boxes, images = self.object_pipeline([datum])
        predicted_labels, predicted_boxes, image = predicted_labels[0], predicted_boxes[0], images[0]

        image_save_path = f"{self.args.visualization_savepath}/images/{datum['image_id']}.png"
        image.save(image_save_path)

        self.visual_explainations[datum['image_id']] = {
            "visual_explaination": image_save_path,
            "text_explaination": ""
        }

        return (predicted_labels, predicted_boxes, datum['image_id'], target_objects)
    
    def ocr(self, image: str, **kwargs):
        results = self.ocr_pipeline.readtext(image)

        boxes = []
        texts = []
        confidences = []

        image_pil = Image.open(image).convert("RGB")
        img = ImageDraw.Draw(image_pil)
        size = image_pil.size

        for result in results:
            box_points, text, conf = result
            text = text.strip().replace(".", "").replace(",", "").replace("?", "").replace("!", "").replace("*", "")

            #xyxy
            box = [ box_points[0][0], box_points[0][1], box_points[-2][0], box_points[-2][1] ]

            try:
                img.rectangle(box, outline="red", width=3)
            except:
                pass

            boxes.append(box)
            texts.append(text.lower().encode("ascii", "ignore").decode())
            confidences.append(conf)

        image_id = self.get_image_id(image)
        image_save_path = f"{self.args.visualization_savepath}/images/{image_id}.png"
        image_pil.save(image_save_path)
        
        self.visual_explainations[image_id] = {
            "visual_explaination": image_save_path,
            "text_explaination": ""
        }

        return (boxes, texts, confidences, self.get_image_id(image))

    def objectEval(self, image, target_object):
        if not self.args.use_obj_vqa:
            predicted_labels, predicted_boxes, image_id, _ = self.objDet(image, target_object)

            correct = len(predicted_labels) > 0
            
            if correct:
                self.visual_explainations[image_id]["text_explaination"] = f"\"{target_object}\" object found."
            else:
                self.visual_explainations[image_id]["text_explaination"] = f"No \"{target_object}\" object found."

            return correct

        question = f"Is there <a> {target_object} in the image?"

        if target_object[0] in [ "a", "e", "i", "o", "u" ]:
            question = question.replace("<a>", "an")
        else:
            question = question.replace("<a>", "a") 

        question = f"Question: {question} Choices: yes, no Answer:"

        image_id = self.get_image_id(image)

        image = self.vis_processors["eval"](Image.open(image).convert("RGB")).unsqueeze(0).to(self.args.device)
        answer = self.model.generate({"image": image, "prompt": question})
        answer = [ a.lower() for a in answer ][0]

        correct = answer == "yes"

        if correct:
            self.visual_explainations[image_id]["text_explaination"] = f"\"{target_object}\" object found."
        else:
            self.visual_explainations[image_id]["text_explaination"] = f"No \"{target_object}\" object found."

        self.visual_explainations[image_id]["visual_explaination"] = f"n/a"

        return correct
    
    def textEval(self, textboxes, target_text: str, **kwargs):
        boxes, texts, confidences, image_id = textboxes

        correct = (target_text.lower() in texts)
        
        if correct:
            self.visual_explainations[image_id]["text_explaination"] = f"\"{target_text}\" text found."
        else:
            self.visual_explainations[image_id]["text_explaination"] = f"No \"{target_text}\" text found."

        return correct
    
    def countEval(self, objects, target_count: str, **kwargs):
        predicted_labels, predicted_boxes, image_id, target_obj = objects

        for count in self.count_map:
            target_count = target_count.replace(count, str(self.count_map[count]))

        if "=" in target_count and not ("==" in target_count):
            if not (">" in target_count and "<" in target_count):
                target_count = f"={target_count}"

        l_dict = { }
        
        try:
            exec(f"x = {len(predicted_labels)}{target_count}", globals(), l_dict)
        except:
            l_dict["x"] = False
        
        x = l_dict["x"]

        if x:
            self.visual_explainations[image_id]["text_explaination"] = f"There are {len(predicted_labels)} \"{target_obj}\" objects."
        else:
            self.visual_explainations[image_id]["text_explaination"] = f"There are {len(predicted_labels)} \"{target_obj}\" objects, not {target_count.replace('=', '').replace('>', '').replace('<', '')}."

        return x
    
    def spatialEval(self, image, objects_relation):

        if not self.args.use_obj_vqa:
            object_1, object_2, relation = objects_relation.split(",")

            if "left" in relation:
                relation = "left"
            elif "right" in relation:
                relation = "right"
            elif "above" in relation:
                relation = "above"
            elif "below" in relation:
                relation = "below"
            elif "front" in relation:
                relation = "front"
            elif "behind" in relation:
                relation = "behind"
            else:
                relation = "skip"

            if not relation == "skip":
                objects = self.objDet(image, f"{object_1},{object_2}")
                correct = self.spatialEvalDino([object_1, object_2], objects, relation)

                if correct:
                    self.visual_explainations[objects[-2]]["text_explaination"] = f"(\"{object_1}\", \"{object_2}\") have the relationship \"{relation}\" between them."
                else:
                    if len(objects[0]) == 0 or not (object_1 in objects[0]):
                        self.visual_explainations[objects[-2]]["text_explaination"] = f"No \"{object_1}\" object found."
                    elif not (object_2 in objects[0]) in objects[0]:
                        self.visual_explainations[objects[-2]]["text_explaination"] = f"No \"{object_2}\" object found."
                    else:
                        self.visual_explainations[objects[-2]]["text_explaination"] = f"No pair of (\"{object_1}\", \"{object_2}\") have the relationship \"{relation}\" between them."

                return correct
            
        question_list = [ ]
                    
        if object_1 == object_2:
            if object_1.endswith(("s", "sh", "ch", "x", "z")):
                obj = f"{object_1}es"
            else:
                obj = f"{object_1}s"

            q1 = f"Are there 2 {obj} in the image?"
        else:
            q1 = f"Is there <a> {object_1} in the image?"
            q2 = f"Is there <a> {object_2} in the image?"

            if object_1[0] in [ "a", "e", "i", "o", "u" ]:
                q1 = q1.replace("<a>", "an")
            else:
                q1 = q1.replace("<a>", "a") 
            if object_2[0] in [ "a", "e", "i", "o", "u" ]:
                q2 = q2.replace("<a>", "an")
            else:
                q2 = q2.replace("<a>", "a") 

            question_list.append(q2)

        question_list.append(q1)
        t = f"Is the {object_1} <tothe>{relation}<of> the {object_2}?"
        
        if relation == "front":
            t = t.replace("<of>", " of")
            t = t.replace("<tothe>", "in ")

        if relation in [ "left", "right" ]:
            t = t.replace("<tothe>", "to the ")
            t = t.replace("<of>", " of")
        else:
            t = t.replace("<tothe>", "")
            t = t.replace("<of>", "")

        question_list.append(t)

        image_id = self.get_image_id(image)

        correct = 0
        image = self.vis_processors["eval"](Image.open(image).convert("RGB")).unsqueeze(0).to(self.args.device)

        self.visual_explainations[image_id] = { }

        for i, q in enumerate(question_list):
            question = f"Question: {q} Choices: yes, no Answer:"

            answer = self.model.generate({"image": image, "prompt": question})
            answer = [ a.lower() for a in answer ][0]
            if answer == "yes":
                correct += 1
            else:
                if i == 0:
                    self.visual_explainations[image_id]["text_explaination"] = f"No \"{object_1}\" object found."
                elif i == 1:
                    self.visual_explainations[image_id]["text_explaination"] = f"No \"{object_2}\" object found."
                else:
                    self.visual_explainations[image_id]["text_explaination"] = f"No pair of (\"{object_1}\", \"{object_2}\") have the relationship \"{relation}\" between them."
                
        correct = correct == len(question_list)
        if correct:
            self.visual_explainations[image_id]["text_explaination"] = f"(\"{object_1}\", \"{object_2}\") have the relationship \"{relation}\" between them."

        self.visual_explainations[image_id]["visual_explaination"] = f"n/a"

        return correct

    def scaleEval(self, image, objects_relation):
        
        if not self.args.use_obj_vqa:
            object_1, object_2, relation = objects_relation.split(",")

            if "small" in relation:
                relation = "smaller"
            elif "big" in relation:
                relation = "bigger"
            elif "same" in relation:
                relation = "same"
            else:
                relation = "skip"

            if not relation == "skip":
                objects = self.objDet(image, f"{object_1},{object_2}")
                correct = self.scaleEvalDino([object_1, object_2], objects, relation)
                
                if correct:
                    self.visual_explainations[objects[-2]]["text_explaination"] = f"(\"{object_1}\", \"{object_2}\") have the relationship \"{relation}\" between them."
                else:
                    if len(objects[0]) == 0 or not (object_1 in objects[0]):
                        self.visual_explainations[objects[-2]]["text_explaination"] = f"No \"{object_1}\" object found."
                    elif not (object_2 in objects[0]) in objects[0]:
                        self.visual_explainations[objects[-2]]["text_explaination"] = f"No \"{object_2}\" object found."
                    else:
                        self.visual_explainations[objects[-2]]["text_explaination"] = f"No pair of (\"{object_1}\", \"{object_2}\") have the relationship \"{relation}\" between them."

                return correct 

        question_list = [ ]
                    
        if object_1 == object_2:
            if object_1.endswith(("s", "sh", "ch", "x", "z")):
                obj = f"{object_1}es"
            else:
                obj = f"{object_1}s"

            q1 = f"Are there 2 {obj} in the image?"
        else:
            q1 = f"Is there <a> {object_1} in the image?"
            q2 = f"Is there <a> {object_2} in the image?"

            if object_1[0] in [ "a", "e", "i", "o", "u" ]:
                q1 = q1.replace("<a>", "an")
            else:
                q1 = q1.replace("<a>", "a") 
            if object_2[0] in [ "a", "e", "i", "o", "u" ]:
                q2 = q2.replace("<a>", "an")
            else:
                q2 = q2.replace("<a>", "a") 

            question_list.append(q2)

        question_list.append(q1)
        
        if relation == "same":
            t = f"Is the {object_1} the same size as the {object_2}?"
        else:
            t = f"Is the {object_1} {relation} than the {object_2}?"

        question_list.append(t)

        image_id = self.get_image_id(image)
        correct = 0
        image = self.vis_processors["eval"](Image.open(image).convert("RGB")).unsqueeze(0).to(self.args.device)
        
        self.visual_explainations[image_id] = { }

        for i, q in enumerate(question_list):
            question = f"Question: {q} Choices: yes, no Answer:"
            answer = self.model.generate({"image": image, "prompt": question})
            answer = [ a.lower() for a in answer ][0]

            if answer == "yes":
                correct += 1
            else:
                if i == 0:
                    self.visual_explainations[image_id]["text_explaination"] = f"No \"{object_1}\" object found."
                elif i == 1:
                    self.visual_explainations[image_id]["text_explaination"] = f"No \"{object_2}\" object found."
                else:
                    self.visual_explainations[image_id]["text_explaination"] = f"No pair of (\"{object_1}\", \"{object_2}\") have the relationship \"{relation}\" between them."
                
        correct = correct == len(question_list)
        if correct:
            self.visual_explainations[image_id]["text_explaination"] = f"(\"{object_1}\", \"{object_2}\") have the relationship \"{relation}\" between them."

        self.visual_explainations[image_id]["visual_explaination"] = f"n/a"

        return correct

    def spatialEvalDino(self, gt_labels, objects, target_relation: str, **kwargs):
        predicted_labels, predicted_boxes, image_id, _ = objects
        
        diff_threshold = 5
        if (len(predicted_labels) >= 2):
            o1sidxs = []
            o2sidxs = []

            if gt_labels[0] != gt_labels[1]:
                for i, label in enumerate(predicted_labels):
                    if label == gt_labels[0]:
                        o1sidxs.append(i)
                    elif label == gt_labels[1]:
                        o2sidxs.append(i)
            else:
                for i, label in enumerate(predicted_labels):
                    if label == gt_labels[0]:
                        o1sidxs.append(i)
                        o2sidxs.append(i)

            center_points = [ ]

            for box in predicted_boxes:
                x, y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                center_points.append((x, y, box[4]))
            
            for o2idx in o2sidxs:
                for o1idx in o1sidxs:
                    o2X, o2Y, o2Z = center_points[o2idx]
                    o1X, o1Y, o1Z = center_points[o1idx]

                    if target_relation == "next":
                        diff = o2X - o1X

                        if diff < -diff_threshold or diff > diff_threshold:
                            return True
                    elif target_relation == "left":
                        diff = o2X - o1X

                        if diff < -diff_threshold:
                            return True
                    elif target_relation == "right":
                        diff = o2X - o1X

                        if diff > diff_threshold:
                            return True
                    elif target_relation == "above":
                        diff = o2Y - o1Y

                        if diff < -diff_threshold:
                            return True
                    elif target_relation == "below":
                        diff = o2Y - o1Y

                        if diff > diff_threshold:
                            return True
                    elif target_relation == "front":
                        if o2Z < o1Z:
                            return True
                    elif target_relation == "behind":
                        if o2Z > o1Z:
                            return True
            
        return False

    def scaleEvalDino(self, gt_labels, objects, target_size: str, **kwargs):
        predicted_labels, predicted_boxes, image_id, _ = objects

        if "small" in target_size:
            target_size = "smaller"
        if "big" in target_size:
            target_size = "bigger"
        if "same" in target_size:
            target_size = "same"

        diff_threshold = 0.05
        if (len(predicted_labels) >= 2):
            o1sidxs = []
            o2sidxs = []

            if gt_labels[0] != gt_labels[1]:
                for i, label in enumerate(predicted_labels):
                    if label == gt_labels[0]:
                        o1sidxs.append(i)
                    elif label == gt_labels[1]:
                        o2sidxs.append(i)
            else:
                for i, label in enumerate(predicted_labels):
                    if label == gt_labels[0]:
                        o1sidxs.append(i)
                        o2sidxs.append(i)

            areas = [ ]

            for box in predicted_boxes:
                area = (box[2] - box[0]) * (box[3] - box[1])
                areas.append(area)
            
            for o2idx in o2sidxs:
                for o1idx in o1sidxs:
                    area2 = areas[o2idx]
                    area1 = areas[o1idx]

                    relative_diff = (area2/area1)
                    
                    if target_size == "same":
                        if abs(1 - relative_diff) < diff_threshold:
                            return True
                    elif target_size == "bigger":
                        if relative_diff > 1 and abs(1 - relative_diff) > diff_threshold:
                            return True
                    elif target_size == "smaller":
                        if relative_diff < 1 and abs(1 - relative_diff) > diff_threshold:
                            return True
            
            return False