import os
import sys
sys.path.append(os.getcwd())

import argparse
import json
import torch
from tqdm import tqdm
import shutil

from evaluation_functions import EvaluationModules

class EvaluationModel:
    def __init__(self, args):
        self.evaluator = EvaluationModules(args)
        self.module_names = EvaluationModules.get_evaluation_functions()

        self.module_weights = {
            "self.evaluator.objectEval": 1,
            "self.evaluator.countEval": 1,
            "self.evaluator.textEval": 1,
            "self.evaluator.spatialEval": 1,
            "self.evaluator.scaleEval": 1,
            "self.evaluator.vqa": 1
        }

        self.module_scores = { }

    def evaluate(self, image_path, model_output):
        for mod in self.module_names:
            model_output = model_output.replace(mod, f"self.evaluator.{mod}")

        image_id = image_path.split("/")[-1].replace(".png", "")
        self.module_scores[image_id] = {}

        correct = 0
        total = 0

        evals = model_output.split(";")

        for i, eval in enumerate(evals):
            # if len(eval) < 2:
            #     continue

            try:
                l_dict = {
                    "image": image_path,
                    "self": self
                }
                module_name = eval.split("(")[0].strip()
                weight = self.module_weights[module_name]
                
                eval = eval.replace("iamge", "image")
                eval = eval.replace("\\\\", "").replace("\\", "")
                eval = eval.replace(" '", " \"").replace("',", "\",").replace("')", "\")")
                
                exec(f"x = {eval}", globals(), l_dict)

                x = l_dict["x"]

                self.module_scores[image_id][eval] = x

                self.evaluator.visual_explainations[f"{image_id}_{i}"] = self.evaluator.visual_explainations[image_id]
                if self.evaluator.visual_explainations[f"{image_id}_{i}"]["visual_explaination"] == "n/a":
                    pass
                else:
                    new_image_path = f"{self.evaluator.args.visualization_savepath}/images/{image_id}_{i}.png"
                    shutil.move(self.evaluator.visual_explainations[f"{image_id}_{i}"]["visual_explaination"], new_image_path)
                    self.evaluator.visual_explainations[f"{image_id}_{i}"]["visual_explaination"] = new_image_path
                if x:
                    correct += weight
                total += weight
            except Exception as ex:
                continue

            del self.evaluator.visual_explainations[f"{image_id}"]
            
        return (correct / total)
    

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser()
    
    ## General Args
    parser.add_argument("--prompt_file", type=str, required=True)

    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--savefile", type=str, default="./results/results.json")
    parser.add_argument("--visualization_savepath", type=str, default="./results/visualizations/")
    parser.add_argument("--device", type=str, default="cuda")
    
    parser.add_argument("--seed", type=int, default=10)

    parser.add_argument("--use_obj_vqa", action="store_true")
    parser.add_argument("--enable_vqa_module", type=str, action="store_true")
    
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

    results = { }

    with open(f"{args.prompt_file}", "r") as f:
        data = json.load(f)
    
    i = 0
    evaluation_model = EvaluationModel(args)
    for id in tqdm(data):
        i += 1
        
        entry = data[id]
        entry["vpeval_score"] = evaluation_model.evaluate(f'{args.image_dir}/{entry["image_path"]}', entry["code"])
        
    with open(f"{args.savefile}", "w") as f:
        json.dump(data, f, indent=2)
        
    with open(f"{args.savefile.replace('.json', '_module_scores.json')}", "w") as f:
        json.dump(evaluation_model.module_scores, f, indent=2)

        
    with open(f"{args.visualization_savepath}/{args.savefile.split('/')[-1].replace('.json', '_explanations.json')}", 'w') as f:
        json.dump(evaluation_model.evaluator.visual_explainations, f, indent=2)