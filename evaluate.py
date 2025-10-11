#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate trained model performance (perplexity)"""
import argparse
import json
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import numpy as np
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True, help="Path to trained model")
    p.add_argument("--base_model", required=True, help="Base model name")
    p.add_argument("--method", choices=["lora", "fullft"], required=True)
    p.add_argument("--dataset", default="openai/gsm8k")
    p.add_argument("--dataset_config", default="main", help="Dataset config")
    p.add_argument("--split", default="test[:500]", help="Test set slice")
    p.add_argument("--max_samples", type=int, default=500)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--bf16", action="store_true", help="Use bfloat16 (should match training)")
    return p.parse_args()


def compute_perplexity(model, tokenizer, texts, max_length=512):
    """Compute perplexity"""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for text in tqdm(texts, desc="Computing perplexity"):
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length
            ).to(model.device)

            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

            total_loss += loss.item() * inputs["input_ids"].size(1)
            total_tokens += inputs["input_ids"].size(1)

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    return perplexity, avg_loss


def main():
    args = parse_args()

    print(f"Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if args.method == "lora":
        # Load LoRA model (must match training dtype)
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            trust_remote_code=True,
            device_map="auto",
            dtype="bfloat16" if args.bf16 else "float16",
        )
        model = PeftModel.from_pretrained(base_model, args.model_path)
        model = model.merge_and_unload()  # Merge LoRA weights
    else:
        # Load full fine-tuned model (must match training dtype)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            device_map="auto",
            dtype="bfloat16" if args.bf16 else "float16",
        )

    print(f"Loading dataset: {args.dataset} ({args.split})")
    if args.dataset_config:
        ds = load_dataset(args.dataset, args.dataset_config, split=args.split)
    else:
        ds = load_dataset(args.dataset, split=args.split)

    # Extract texts (handle different dataset formats)
    texts = []
    for item in ds:
        # GSM8K format: question + answer
        if 'question' in item and 'answer' in item:
            texts.append(f"Question: {item['question']}\nAnswer: {item['answer']}")

        # Conversations format
        elif 'conversations' in item:
            conv_text = ""
            for msg in item['conversations']:
                if isinstance(msg, dict) and 'value' in msg:
                    conv_text += msg['value'] + " "
            if conv_text:
                texts.append(conv_text.strip())

        # Plain text format
        elif 'text' in item and item['text']:
            texts.append(item['text'])

        if len(texts) >= args.max_samples:
            break

    print(f"Evaluating {len(texts)} samples")

    # Compute perplexity
    perplexity, avg_loss = compute_perplexity(model, tokenizer, texts, args.max_length)

    results = {
        "method": args.method,
        "model_path": args.model_path,
        "perplexity": round(perplexity, 4),
        "avg_loss": round(avg_loss, 4),
        "num_samples": len(texts),
    }

    # Save results
    output_file = f"{args.model_path}/eval_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Evaluation Results ({args.method}):")
    print(f"  Perplexity: {perplexity:.4f}")
    print(f"  Average Loss: {avg_loss:.4f}")
    print(f"  Results saved to: {output_file}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()


# Usage examples:
# python3 evaluate.py --model_path ./lora-sft --base_model meta-llama/Llama-3.2-1B-Instruct --method lora --dataset openai/gsm8k --dataset_config main --split "test[:500]" --bf16
# python3 evaluate.py --model_path ./fullft-sft --base_model meta-llama/Llama-3.2-1B-Instruct --method fullft --dataset openai/gsm8k --dataset_config main --split "test[:500]" --bf16
