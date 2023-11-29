# Changelog

## Nov 29, 2023
- We've released an end-to-end inference notebook, `inference.ipynb` or<a target="_blank" href="https://colab.research.google.com/github/aszala/VPEval/blob/main/inference.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

- We've released a fine-tuned (on ChatGPT outputs) LLama2 7B model.
If you do not want to use ChatGPT then you can use this model instead.
Please refer to this [code file](https://github.com/aszala/VPEval/blob/main/src/open_ended_prompt_generation/local_llm_program_generation/llama2_generation.py). 

## Aug 20, 2023

  - We add an experiment with Vicuna13B+GLIGEN trained only on the "Flickr30K" dataset and compare it to Vicuna13B+GLIGEN trained on the "Flickr30K+COCO+PaintSkills" datasets (i.e. to compare training on only real images vs real+simulated images).
  - We update the evaluation script for spatial/scale skills of VPEval (for the spatial+scale object ordering). This improves the human correlation of VPEval in both skill-based (66.6 -> **73.5**; Table 3) and open-ended (56.2 -> **56.9**; Table 4) prompts. Also, we provide below the updated T2I evaluation results (i.e. Tables 1 and 2 in the paper) with the updated evaluation script (where VPGen improves on both spatial+scale skills). We will also update these changes in the next arXiv version.

Table 1. VPEval (skill-based prompts)

|                                     | Spatial | Scale | Average |
|-------------------------------------|:-------:|:-----:|:-------:|
| Stable Diffusion v1.4               |    23   |   12  |   37.7  |
| Stable Diffusion v2.1               |    31   |   14  |   40.6  |
| Karlo                               |    24   |   16  |   40.8  |
| minDALL-E                           |    7    |   6   |   24.4  |
| DALL-E Mega                         |    17   |   9   |   33.0  |
| Vicuna13B+GLIGEN (Flickr30K+COCO+PaintSkills)  | **56**  |  **26** |  **51.0**  |
| Vicuna13B+GLIGEN (Flickr30K)           |    39   |   23  |   43.8  |

<br>

Table 2. VPEval (open-ended prompts)

|                                     | Score (%) |
|-------------------------------------|:---------:|
| Stable Diffusion v1.4               |    70.6   |
| Stable Diffusion v2.1               |    72.0   |
| Karlo                               |    70.0   |
| minDALL-E                           |    47.5   |
| DALL-E Mega                         |    67.2   |
| Vicuna13B+GLIGEN (Flickr30K+COCO+PaintSkills) | 68.3 |
| Vicuna13B+GLIGEN (Flickr30K)            |    71.0   |