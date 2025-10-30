#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate trained Llama-3.1-8B models on GSM8K test set
"""
import os
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "offline"

import argparse
import json
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True, help="Path to trained model")
    p.add_argument("--base_model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--method", required=True, choices=["lora-attention", "lora-all", "fullft"])
    p.add_argument("--dataset", default="openai/gsm8k")
    p.add_argument("--dataset_config", default="main")
    p.add_argument("--split", default="test")
    p.add_argument("--max_samples", type=int, default=500, help="Max samples to evaluate")
    p.add_argument("--bf16", action="store_true")
    return p.parse_args()


def compute_perplexity(model, tokenizer, dataset, max_samples=500):
    """Compute perplexity on the dataset"""
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i, example in enumerate(tqdm(dataset, desc="Evaluating", total=min(len(dataset), max_samples))):
            if i >= max_samples:
                break

            # Format as chat message
            messages = [
                {"role": "user", "content": example["question"]},
                {"role": "assistant", "content": example["answer"]}
            ]

            # Apply chat template
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )

            # Tokenize
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Compute loss
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

            # Accumulate
            num_tokens = inputs["input_ids"].size(1)
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    # Compute perplexity
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)

    return perplexity, avg_loss


def main():
    args = parse_args()

    print("="*60)
    print(f"Evaluating {args.method.upper()}")
    print("="*60)
    print()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    print("✓ Tokenizer loaded")
    print()

    # Load model
    print(f"Loading model from {args.model_path}...")
    dtype = torch.bfloat16 if args.bf16 else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=dtype,
    )
    print("✓ Model loaded")
    print()

    # Load test dataset
    print("Loading test dataset...")
    test_dataset = load_dataset(args.dataset, args.dataset_config, split=args.split)
    print(f"✓ Test dataset loaded: {len(test_dataset)} samples")
    print(f"  Evaluating on first {args.max_samples} samples")
    print()

    # Compute perplexity
    print("Computing perplexity...")
    perplexity, avg_loss = compute_perplexity(model, tokenizer, test_dataset, args.max_samples)

    print()
    print("="*60)
    print("Evaluation Results:")
    print(f"  Perplexity: {perplexity:.4f}")
    print(f"  Average Loss: {avg_loss:.4f}")
    print("="*60)
    print()

    # Save results
    results = {
        "method": args.method,
        "perplexity": float(perplexity),
        "avg_loss": float(avg_loss),
        "num_samples": min(len(test_dataset), args.max_samples),
    }

    output_path = os.path.join(args.model_path, "eval_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✓ Results saved to: {output_path}")
    print()


if __name__ == "__main__":
    main()
