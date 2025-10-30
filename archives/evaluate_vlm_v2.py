#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate trained VLM model performance - CORRECTED VERSION
"""
import argparse
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,  # ✅ Correct class
    BitsAndBytesConfig
)
from peft import PeftModel
import numpy as np
from tqdm import tqdm
import re


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate VLM models (CORRECTED)")
    p.add_argument("--model_path", required=True, help="Path to trained model")
    p.add_argument("--base_model", default="llava-hf/llava-1.5-7b-hf", help="Base VLM model")
    p.add_argument("--method", choices=["lora-attention", "lora-all", "fullft"], required=True)

    # Dataset
    p.add_argument("--dataset", default="derek-thomas/ScienceQA")
    p.add_argument("--dataset_config", default=None)
    p.add_argument("--split", default="test[:500]")
    p.add_argument("--max_samples", type=int, default=500)

    # Generation
    p.add_argument("--max_new_tokens", type=int, default=10)

    # Precision
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--use_4bit", action="store_true")

    return p.parse_args()


def extract_answer_letter(text):
    """Extract answer letter (A/B/C/D/E) from generated text"""
    text = text.strip().upper()

    # Pattern 1: Direct letter
    if len(text) > 0 and text[0] in ['A', 'B', 'C', 'D', 'E']:
        return text[0]

    # Pattern 2: Letter in text
    match = re.search(r'\b([A-E])\b', text)
    if match:
        return match.group(1)

    return None


def evaluate_scienceqa(model, processor, dataset, args):
    """Evaluate on ScienceQA"""
    print(f"\nEvaluating on ScienceQA...")

    correct = 0
    total = 0
    predictions = []

    model.eval()

    with torch.no_grad():
        for idx, item in enumerate(tqdm(dataset, desc="Evaluating")):
            if total >= args.max_samples:
                break

            # Skip samples without images
            if item['image'] is None:
                continue

            # Format question
            question = item['question']
            choices = item['choices']
            answer_idx = item['answer']

            choices_text = "\n".join([f"{chr(65+i)}. {choice}"
                                     for i, choice in enumerate(choices)])
            user_content = f"Question: {question}\n\nChoices:\n{choices_text}\n\nAnswer:"

            # ✅ CRITICAL: Use conversation format with image token placeholder
            # LLaVA expects a specific format with <image> token
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},  # This will be replaced with <image> token
                        {"type": "text", "text": user_content}
                    ]
                }
            ]

            # Apply chat template to get proper prompt format
            prompt = processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False
            )

            # Prepare inputs
            inputs = processor(
                text=prompt,
                images=item['image'],
                return_tensors="pt"
            ).to(model.device)

            # Generate
            try:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.eos_token_id
                )

                # Decode
                generated_ids = outputs[0][inputs.input_ids.shape[1]:]
                generated_text = processor.decode(generated_ids, skip_special_tokens=True).strip()

                # Extract answer
                predicted_letter = extract_answer_letter(generated_text)
                correct_letter = chr(65 + answer_idx)

                is_correct = (predicted_letter == correct_letter)
                if is_correct:
                    correct += 1
                total += 1

                # Log
                predictions.append({
                    "question": question,
                    "choices": choices,
                    "correct_answer": correct_letter,
                    "predicted_answer": predicted_letter,
                    "generated_text": generated_text,
                    "is_correct": is_correct,
                })

                # Print sample results
                if idx < 5:
                    print(f"\n--- Sample {idx+1} ---")
                    print(f"Generated: {generated_text}")
                    print(f"Predicted: {predicted_letter}, Correct: {correct_letter} {'✓' if is_correct else '✗'}")

            except Exception as e:
                print(f"\nError on sample {idx}: {e}")
                continue

    accuracy = correct / total if total > 0 else 0

    results = {
        "method": args.method,
        "model_path": args.model_path,
        "dataset": args.dataset,
        "eval_type": "accuracy",
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": total,
        "predictions": predictions[:100],
    }

    return results, accuracy, correct, total


def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print("VLM Evaluation - CORRECTED VERSION")
    print(f"{'='*60}\n")
    print(f"Loading model from: {args.model_path}")

    # Load processor
    processor = AutoProcessor.from_pretrained(args.model_path)

    # Configure quantization
    quantization_config = None
    if args.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

    # Load model
    if args.method in ["lora-attention", "lora-all"]:
        print(f"Loading base model: {args.base_model}")
        base_model = AutoModelForImageTextToText.from_pretrained(
            args.base_model,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16 if args.bf16 and not args.use_4bit else torch.float16,
            trust_remote_code=True,
        )

        print(f"Loading LoRA adapter: {args.model_path}")
        model = PeftModel.from_pretrained(base_model, args.model_path)
        print("Merging LoRA weights...")
        model = model.merge_and_unload()

    else:
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_path,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16 if args.bf16 and not args.use_4bit else torch.float16,
            trust_remote_code=True,
        )

    # Load dataset
    print(f"\nLoading dataset: {args.dataset} ({args.split})")
    if args.dataset_config:
        ds = load_dataset(args.dataset, args.dataset_config, split=args.split)
    else:
        ds = load_dataset(args.dataset, split=args.split)

    # Filter
    if 'image' in ds.column_names:
        original_size = len(ds)
        ds = ds.filter(lambda x: x['image'] is not None)
        print(f"Filtered: {original_size} → {len(ds)} samples (with images)")

    # Evaluate
    if "scienceqa" in args.dataset.lower():
        results, accuracy, correct, total = evaluate_scienceqa(model, processor, ds, args)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # Print results
    print(f"\n{'='*60}")
    print(f"Evaluation Results ({args.method}):")
    print(f"  Dataset: {args.dataset}")
    print(f"  Accuracy: {accuracy:.2%} ({correct}/{total})")
    print(f"{'='*60}\n")

    # Save
    output_file = f"{args.model_path}/eval_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✓ Results saved to: {output_file}")


if __name__ == "__main__":
    main()
