import os
import sys
sys.path.append(os.getcwd())

import argparse
import json
import torch
from tqdm import tqdm

from evaluation_functions import EvaluationModules

def parse_metadata(args, skill):
    with open(f"{args.metadata_path}/random_ids_{skill}.json", 'r') as f:
        safe_ids = json.load(f)[:1000]

    with open(f"{args.metadata_path}/{skill}_new.json", 'r') as f:
        metadata: list[dict] = json.load(f)["data"]

    data: list[dict] = [ ]

    for entry in metadata:
        prompt: str = entry["text"]
        image_id: str = entry["id"]
        gt_labels: list[str] = [ ]

        if not (image_id in safe_ids):
            continue

        for object in entry["objects"]:
            gt_label: str = object["coconame"]

            gt_labels.append(gt_label)

        skill = image_id.split("_")[0]

        if skill == "object":
            target_object = gt_labels[0]
            
            code = f"objectEval(image, '{target_object}')"
        elif skill == "count":
            target_object = gt_labels[0]
            target_count = len(gt_labels)

            code = f"countEval(objDet(image, '{target_object}'), '=={target_count}')"
        elif skill == "spatial":
            target_object_1 = gt_labels[0]
            target_object_2 = gt_labels[1]
            relation = entry['objects'][-1]['relation'].split('_')[0]

            code = f"spatialEval(image, '{target_object_1},{target_object_2},{relation}')"
        elif skill == "scale":
            target_object_1 = gt_labels[0]
            target_object_2 = gt_labels[1]
            relation = entry['objects'][-1]['scale']

            code = f"scaleEval(image, '{target_object_1},{target_object_2},{relation}')"
        elif skill == "text":
            text = prompt.split("'")[1]

            code = f"textEval(ocr(image), '{text}')"


        datum: dict = {
            "prompt": prompt,
            "image_id": image_id,
            "code": code
        }

        if skill == "spatial":
            datum["sub"] = entry["objects"][-1]["relation"].split("_")[0]
        elif skill == "scale":
            datum["sub"] = entry["objects"][-1]["scale"]
        elif skill == "count":
            datum["sub"] = str(len(gt_labels))

        data.append(datum)

    return data

class EvaluationModel:
    def __init__(self, args):
        self.evaluator = EvaluationModules(args)
        self.module_names = EvaluationModules.get_evaluation_functions()

    def evaluate(self, image_path, eval_code):
        for mod in self.module_names:
            eval_code = eval_code.replace(mod, f"self.evaluator.{mod}")

        l_dict = {
            "image": image_path,
            "self": self
        }

        exec(f"x = {eval_code}", globals(), l_dict)
        
        x = l_dict["x"]

        return x
    

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser()
    
    ## General Args
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--metadata_path", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--savepath", type=str, default="./results/")
    parser.add_argument("--visualization_savepath", type=str, default="./results/visualizations/")
    parser.add_argument("--skills", type=str, default="object,count,spatial,scale,text")
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--enable_vqa_module", type=str, action="store_true")
    parser.add_argument("--use_obj_vqa", action="store_true")

    parser.add_argument("--seed", type=int, default=0)

    ## GroundingDino Args
    parser.add_argument("--grounding_dino_config_path", type=str, default="./src/dino/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--grounding_dino_weights_path", type=str, default="./weights/groundingdino_swint_ogc.pth")

    ## EasyOCR Args
    parser.add_argument("--ocr_langauge", type=str, default="en")

    args = parser.parse_args()

    from accelerate.utils import set_seed
    import random
    import numpy as np
    
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    save_path = args.savepath
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    skills = args.skills.split(",")

    skill_sub_skills = {
        "spatial": [ "left", "right", "below", "above", "front", "behind" ],
        "count": [ "1", "2", "3", "4" ],
        "scale": [ "bigger", "smaller", "same" ]
    }

    results = { }
        
    evaluation_model = EvaluationModel(args)

    for skill in tqdm(skills, colour="green"):
        data = parse_metadata(args, skill)
        results[skill] = {
            "accuracy": 0,
            "total": 0,
            "correct": 0,
            "incorrect": 0,
            "lm_input_fails": 0,
            "sub_skills": { }
        }

        if skill in skill_sub_skills:
            for sub in skill_sub_skills[skill]:
                results[skill]["sub_skills"][sub] = { 
                    "accuracy": 0,
                    "total": 0,
                    "correct": 0,
                    "incorrect": 0,
                }

        for i, d in tqdm(enumerate(data), total=len(data)):
            x = evaluation_model.evaluate(f"{args.image_dir}/{args.model}/{skill}/{d['image_id']}.png", d['code'])

            sub = ""
            if "sub" in d:
                sub = d["sub"]

            results[skill]["total"] += 1
            if len(sub) > 0:
                results[skill]["sub_skills"][sub]["total"] += 1
            if x:
                results[skill]["correct"] += 1
                if len(sub) > 0:
                    results[skill]["sub_skills"][sub]["correct"] += 1
            else:
                results[skill]["incorrect"] += 1
                if len(sub) > 0:
                    results[skill]["sub_skills"][sub]["incorrect"] += 1

        results[skill]["accuracy"] = results[skill]["correct"] / results[skill]["total"]

        for sub in results[skill]["sub_skills"]:
            if results[skill]["sub_skills"][sub]["total"] > 0:
                results[skill]["sub_skills"][sub]["accuracy"] = results[skill]["sub_skills"][sub]["correct"] / results[skill]["sub_skills"][sub]["total"]


    with open(f"{save_path}{args.model}_skill_based.json", 'w') as f:
        json.dump(results, f, indent=2)

    with open(f"{args.visualization_savepath}/{args.model}_skill_based_explanations.json", 'w') as f:
        json.dump(evaluation_model.evaluator.visual_explainations, f, indent=2)