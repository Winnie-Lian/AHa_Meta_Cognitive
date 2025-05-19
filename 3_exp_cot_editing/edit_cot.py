import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import json
from tqdm import tqdm


model_path = "<your_path_of_DeepSeek>"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

template_path = Path("./think_template.jinjia")
chat_template = template_path.read_text(encoding="utf-8")
tokenizer.chat_template = chat_template


def extract_think_and_answer(text):
    start_tag = "<think>"
    end_tag = "</think>"
    start_idx = text.find(start_tag)
    end_idx = text.find(end_tag)
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        think_content = text[start_idx + len(start_tag):end_idx].strip()
        answer_content = text[end_idx + len(end_tag):].strip()
    else:
        think_content = ""
        answer_content = ""
    return think_content, answer_content


def get_modified_response(query_cot):
    conversation = [{"role": "user", "content": query_cot}]
    input_text = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10240,
            do_sample=False,

        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    input_file = "./correct_ha_14B_QA_modified.json"
    output_file = "./correct_ha_14B_QA_modified-result.json"

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    modified_data = []

    for item in tqdm(data, desc="Processing questions"):
        question = item["question"]
        modify_cots = item.get("modify", [])

        for i in modify_cots:
            modify_cot = i["modified_cot"]
            query = f"{question}<think>{modify_cot}"
            try:
                output = get_modified_response(query)
                new_cot, new_answer = extract_think_and_answer(output)
            except Exception as e:
                print(f"Error processing item: {question}\n{e}")
                new_cot, new_answer = "", ""
            i["output"]=output
            i["new_cot"] = new_cot
            i["new_answer"] = new_answer

        modified_data.append(item)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(modified_data, f, ensure_ascii=False, indent=2)

    print(f"Output saved to {output_file}")
    
if __name__ == "__main__":
    main()
