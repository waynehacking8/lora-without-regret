#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import time
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, EarlyStoppingCallback
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig


class MetricsCallback(TrainerCallback):
    """Track training metrics: loss, GPU memory, training time"""
    def __init__(self):
        self.metrics_log = []
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            gpu_mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            elapsed = time.time() - self.start_time

            metric = {
                "step": state.global_step,
                "loss": logs.get("loss"),
                "learning_rate": logs.get("learning_rate"),
                "gpu_memory_gb": round(gpu_mem, 2),
                "elapsed_time_min": round(elapsed / 60, 2),
            }
            self.metrics_log.append(metric)

    def save_metrics(self, output_dir, method_name):
        filepath = f"{output_dir}/metrics_{method_name}.json"
        with open(filepath, "w") as f:
            json.dump(self.metrics_log, f, indent=2)
        print(f"✓ Metrics saved to: {filepath}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", default="openai/gsm8k")
    p.add_argument("--dataset_config", default="main", help="Dataset config (e.g., 'main' for GSM8K)")
    p.add_argument("--split", default="train", help="Training dataset split")
    p.add_argument("--eval_split", default="test[:500]", help="Validation dataset split (for early stopping)")
    p.add_argument("--out", default="./out-sft")
    p.add_argument("--method", choices=["lora", "fullft"], required=True)
    p.add_argument("--num_train_epochs", type=int, default=15, help="Number of training epochs (default: 15)")
    p.add_argument("--max_steps", type=int, default=-1, help="Max steps (overrides epochs if > 0)")
    p.add_argument("--max_seq_length", type=int, default=1024)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--early_stopping_patience", type=int, default=3, help="Early stopping patience (epochs)")
    p.add_argument("--lora_rank", type=int, default=256, help="LoRA rank (HF SFT: 256)")
    p.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha (HF SFT: 16)")
    p.add_argument("--lora_lr", type=float, default=2e-4, help="LoRA LR (HF SFT example: 2e-4)")
    p.add_argument("--fullft_lr", type=float, default=2e-5, help="Full FT LR (inferred 10x lower)")
    return p.parse_args()


def load_model_tokenizer(args):
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tok.pad_token = tok.eos_token if tok.pad_token is None else tok.pad_token

    # Fair comparison: Both LoRA and Full FT use the same dtype (following HF docs)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        device_map="auto",
        dtype="bfloat16" if args.bf16 else "float16",
    )
    # Note: SFTConfig defaults to gradient_checkpointing=True, automatically enabled

    return model, tok


def main():
    a = parse_args()
    model, tok = load_model_tokenizer(a)

    # Load training dataset
    if a.dataset_config:
        train_ds = load_dataset(a.dataset, a.dataset_config, split=a.split)
        eval_ds = load_dataset(a.dataset, a.dataset_config, split=a.eval_split)
    else:
        train_ds = load_dataset(a.dataset, split=a.split)
        eval_ds = load_dataset(a.dataset, split=a.eval_split)

    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(eval_ds)}")

    # Convert dataset to messages format for SFTTrainer
    def convert_to_messages(example):
        """Convert various dataset formats to 'messages' format"""

        # GSM8K format: question + answer
        if 'question' in example and 'answer' in example:
            messages = [
                {"role": "user", "content": example['question']},
                {"role": "assistant", "content": example['answer']}
            ]
            return {"messages": messages}

        # OpenThoughts format: conversations
        elif 'conversations' in example:
            messages = []
            for msg in example['conversations']:
                role = "user" if msg['from'] in ['user', 'human'] else "assistant"
                messages.append({"role": role, "content": msg['value']})
            return {"messages": messages}

        # Already in messages format
        elif 'messages' in example:
            return example

        # Unsupported format
        else:
            raise ValueError(f"Unsupported dataset format. Available keys: {example.keys()}")

    # Apply conversion to both datasets
    original_columns = train_ds.column_names
    train_ds = train_ds.map(convert_to_messages, remove_columns=[col for col in original_columns if col != 'messages'])
    eval_ds = eval_ds.map(convert_to_messages, remove_columns=[col for col in original_columns if col != 'messages'])

    peft_cfg = None
    # LoRA without Regret: HF docs specific values
    if a.method == "lora":
        lr = a.lora_lr  # 1e-5 per HF docs
        peft_cfg = LoraConfig(
            r=a.lora_rank,
            lora_alpha=a.lora_alpha,
            lora_dropout=0.0,  # HF docs: use 0.0
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )
    else:
        lr = a.fullft_lr  # 1e-6 per HF docs

    # Use epochs by default, or max_steps if specified
    training_args = {
        "output_dir": a.out,
        "per_device_train_batch_size": a.per_device_train_batch_size,
        "per_device_eval_batch_size": a.per_device_train_batch_size,
        "gradient_accumulation_steps": a.grad_accum,
        "learning_rate": lr,
        "logging_steps": 10,
        "bf16": a.bf16,
        "fp16": not a.bf16,
        "max_seq_length": a.max_seq_length,
        "packing": True,
        # Evaluation and checkpoint strategy
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "save_total_limit": 3,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
    }

    # Use max_steps if specified, otherwise use epochs
    if a.max_steps > 0:
        training_args["max_steps"] = a.max_steps
    else:
        training_args["num_train_epochs"] = a.num_train_epochs

    cfg = SFTConfig(**training_args)

    metrics_cb = MetricsCallback()
    early_stopping_cb = EarlyStoppingCallback(early_stopping_patience=a.early_stopping_patience)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        peft_config=peft_cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=cfg,
        callbacks=[metrics_cb, early_stopping_cb],
    )

    print(f"\n{'='*60}")
    print(f"Training Configuration:")
    print(f"  Method: {a.method.upper()}")
    print(f"  Learning Rate: {lr}")
    print(f"  Epochs: {a.num_train_epochs if a.max_steps <= 0 else f'max_steps={a.max_steps}'}")
    print(f"  Early Stopping Patience: {a.early_stopping_patience} epochs")
    print(f"  Batch Size: {a.per_device_train_batch_size} x {a.grad_accum} = {a.per_device_train_batch_size * a.grad_accum}")
    print(f"{'='*60}\n")

    trainer.train()
    trainer.save_model()
    tok.save_pretrained(a.out)
    metrics_cb.save_metrics(a.out, a.method)


if __name__ == "__main__":
    main()


# Usage examples:

# LoRA without Regret with GSM8K
# python3 sft_compare.py \
#   --model meta-llama/Llama-3.2-1B-Instruct \
#   --dataset openai/gsm8k --dataset_config main --split train \
#   --method lora --out ./lora-sft --bf16 \
#   --per_device_train_batch_size 1 --grad_accum 4 --max_steps 1500

# Full Fine-tuning with GSM8K
# python3 sft_compare.py \
#   --model meta-llama/Llama-3.2-1B-Instruct \
#   --dataset openai/gsm8k --dataset_config main --split train \
#   --method fullft --out ./fullft-sft --bf16 \
#   --per_device_train_batch_size 1 --grad_accum 4 --max_steps 1500