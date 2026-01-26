import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID ="mistralai/Mistral-7B-Instruct-v0.3"

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map = "auto",
    )

    prompt = "Write one sentence explaining what evaluatuion driven posttraining means."
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs,
            max_new_tokens = 60,
            do_sample = True,
            temperature = 0.7,
            top_p = 0.9
            )

    text = tokenizer.decode(outputs[0], skip_special_tokens = True)
    print(text)

if __name__ == "__main__":
    main()