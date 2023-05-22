import json
import numpy as np


for model in [ "sd_14", "sd_21", "karlo", "mindalle", "dallemega", "vpgen_gligen" ]:
    
    final_scores = [ ]
    for dataset in [ "paintskill", "coco", "parti", "drawbench" ]:

        with open(f"./results/{model}_{dataset}.json", 'r') as f:
            data = json.load(f)
            
        scores = [ ]

        for id in data:
            entry = data[id]
            score = entry["vpeval_score"]
            scores.append(score)

        scores = np.array(scores)
        mean = np.mean(scores) * 100
        print(f"{model} {dataset}: {mean}")
        final_scores.append(mean)

    print(f"{model} overall average: {np.mean(final_scores)}")