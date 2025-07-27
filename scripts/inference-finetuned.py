import os
import re
import json
import argparse
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer


def detect_device():
    if torch.cuda.is_available():
        return "cuda"
    try:
        if torch.backends.mps.is_available():
            return "mps"
    except: # noqa: E722
        pass
    return "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, required=True, help="Path or name of the base model")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to dataset")
    parser.add_argument('--save_dir', type=str, required=True, help="Directory to save the output")
    parser.add_argument('--save_name', type=str, required=True, help="File to save the output")
    parser.add_argument('--load_8bit', action='store_true', help="Use 8-bit loading")
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--max_length', type=int, default=512)


    args = parser.parse_args()
    device = detect_device()
    print(f"Running on device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, cache_dir=args.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        attn_implementation="eager",
        torch_dtype='auto',
        cache_dir=args.cache_dir
    )
    model.eval()

    def evaluate(prompt,
                 temperature=0.1,
                 max_new_tokens=128,):
        model_inputs = tokenizer(prompt, return_tensors="pt").to(device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        # Calculate input length for slicing
        input_length = model_inputs.input_ids.shape[1]

        # Slice generated output to exclude the prompt
        generated_ids = [output_ids[input_length:] for output_ids in generated_ids]

        responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        return responses

    # Load test data
    data = load_dataset("json", data_files=args.dataset_path)
    samples = data['train']

    os.makedirs(os.path.dirname(args.save_dir), exist_ok=True)
    save_path = os.path.join(args.save_dir, args.save_name)

    with open(save_path, 'a') as f:
        for i, sample in enumerate(tqdm(samples)):
            input_prompt = f"{sample['input']}\n\n"
            reason_answer = evaluate(input_prompt, max_new_tokens=args.max_length, temperature=args.temperature)
            print(reason_answer)
            print("-"*30)

            output = {'id': sample['id'],
                      'input': sample['input'],
                      'label': sample['label'],
                      'fol': reason_answer
                      }
            f.write(json.dumps(output, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    main()
