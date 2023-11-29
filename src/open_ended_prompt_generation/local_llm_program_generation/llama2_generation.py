import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


model_name = "abhayzala/vpeval-program-generation-llama-2-7b"

tokenizer = AutoTokenizer.from_pretrained(model_name)

pipeline = transformers.pipeline(
    "text-generation",
    model=model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)


# format dataset. Follow LLaMA 2 style
def create_qg_prompt(caption):

    INTRO_BLURB = """Given an image description, generate programs that verify if the related image is correct.
"""
    
    formated_prompt = f"<s>[INST] <<SYS>>\n{INTRO_BLURB}\n<</SYS>>\n\n"
    formated_prompt += f"Description: {caption} [/INST] "
    return formated_prompt

def question_generation(caption):
    
    prompt = create_qg_prompt(caption)
    
    sequences = pipeline(prompt, do_sample=False, num_beams=5, num_return_sequences=1, max_length=512)
    
    output = sequences[0]['generated_text'][len(prompt):]
    output = output.split('\n\n')[0]
    return output


if __name__ == "__main__":
    
    test_caption_1 = "a blue rabbit and a red plane"
    print(test_caption_1)
    print(question_generation(test_caption_1))
    print('-------------------'*10)
    
    test_caption_2 = "a bear that is to the left of a tree"
    print(test_caption_2)
    print(question_generation(test_caption_2))
    print('-------------------'*10)
    
    test_caption_3 = "three bears next to a tree"
    print(test_caption_3)
    print(question_generation(test_caption_3))
    print('-------------------'*10)