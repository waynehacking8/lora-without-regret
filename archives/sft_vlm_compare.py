#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA without Regret for Vision-Language Models (VLMs)
Training script for LLaVA with LoRA-Attention, LoRA-All, and Full Fine-Tuning
"""
import argparse
import json
import time
import torch
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    TrainerCallback,
    EarlyStoppingCallback,
    BitsAndBytesConfig
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model
import os


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
    p = argparse.ArgumentParser(description="VLM LoRA without Regret experiments")

    # Model arguments
    p.add_argument("--model", default="llava-hf/llava-1.5-7b-hf", help="VLM model name")
    p.add_argument("--method", choices=["lora-attention", "lora-all", "fullft"], required=True)

    # Dataset arguments
    p.add_argument("--dataset", default="derek-thomas/ScienceQA", help="Dataset name")
    p.add_argument("--dataset_config", default=None, help="Dataset config (if applicable)")
    p.add_argument("--split", default="train", help="Training dataset split")
    p.add_argument("--eval_split", default="validation[:500]", help="Validation split")
    p.add_argument("--max_samples", type=int, default=-1, help="Max training samples (-1 for all)")

    # Output
    p.add_argument("--out", default="./out-vlm-sft", help="Output directory")

    # Training hyperparameters
    p.add_argument("--num_train_epochs", type=int, default=5, help="Number of training epochs")
    p.add_argument("--max_steps", type=int, default=-1, help="Max steps (overrides epochs if > 0)")
    p.add_argument("--max_seq_length", type=int, default=2048, help="Max sequence length (VLM needs longer)")
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--early_stopping_patience", type=int, default=3, help="Early stopping patience")

    # LoRA configuration (following LoRA without Regret)
    p.add_argument("--lora_rank", type=int, default=256, help="LoRA rank (default: 256)")
    p.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha (default: 16)")
    p.add_argument("--lora_lr", type=float, default=1e-4, help="LoRA learning rate (7B model: 1e-4)")
    p.add_argument("--fullft_lr", type=float, default=1e-5, help="Full FT learning rate (10x lower)")

    # Precision
    p.add_argument("--bf16", action="store_true", help="Use bfloat16 precision")
    p.add_argument("--use_4bit", action="store_true", help="Use 4-bit quantization (for memory efficiency)")

    # Vision encoder
    p.add_argument("--freeze_vision_tower", action="store_true", default=True,
                   help="Freeze vision encoder (recommended for fair comparison)")

    return p.parse_args()


def load_model_processor(args):
    """Load VLM model and processor"""

    print(f"Loading model: {args.model}")
    print(f"  Method: {args.method}")
    print(f"  Freeze vision tower: {args.freeze_vision_tower}")
    print(f"  Use 4-bit: {args.use_4bit}")

    # Load processor (handles both image and text)
    processor = AutoProcessor.from_pretrained(args.model)

    # Configure quantization if needed
    quantization_config = None
    if args.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

    # Load model
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if args.bf16 and not args.use_4bit else torch.float16,
    )

    # Freeze vision tower if specified (recommended for fair comparison with LLM experiments)
    if args.freeze_vision_tower:
        for param in model.vision_tower.parameters():
            param.requires_grad = False
        print("✓ Vision tower frozen")

    return model, processor


def prepare_scienceqa_dataset(examples, processor):
    """
    Prepare ScienceQA dataset for VLM training
    Format: Image + Question + Choices → Answer
    """
    texts = []
    images = []

    for i in range(len(examples['image'])):
        # Skip samples without images
        if examples['image'][i] is None:
            continue

        # Format question with choices
        question = examples['question'][i]
        choices = examples['choices'][i]
        answer_idx = examples['answer'][i]

        # Build choices text
        choices_text = "\n".join([f"{chr(65+j)}. {choice}" for j, choice in enumerate(choices)])

        # Build prompt
        prompt = f"Question: {question}\n\nChoices:\n{choices_text}\n\nAnswer:"

        # Answer
        answer_letter = chr(65 + answer_idx)
        answer = f" {answer_letter}"

        # Combine
        full_text = prompt + answer

        texts.append(full_text)
        images.append(examples['image'][i])

    # Process with processor
    model_inputs = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    # Add labels for training (same as input_ids)
    model_inputs["labels"] = model_inputs["input_ids"].clone()

    return model_inputs


def get_lora_config(args):
    """Configure LoRA based on method"""

    if args.method == "lora-attention":
        # LoRA only on language model attention layers
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj"
        ]
        print(f"LoRA-Attention: {len(target_modules)} modules (LLM attention only)")

    elif args.method == "lora-all":
        # LoRA without Regret: all linear layers in language model
        # Note: We keep vision tower frozen, so we only apply to language_model
        target_modules = [
            # Attention
            "q_proj", "k_proj", "v_proj", "o_proj",
            # MLP
            "gate_proj", "up_proj", "down_proj",
        ]
        print(f"LoRA-All: {len(target_modules)} modules (all LLM linear layers)")

    else:
        return None  # Full fine-tuning

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
        # Apply only to language_model (not vision_tower or multi_modal_projector)
        modules_to_save=None,
    )

    return lora_config


def collate_fn(examples, processor):
    """Custom collate function for VLM"""
    # Extract texts and images
    texts = [ex['text'] if 'text' in ex else ex['input_ids'] for ex in examples]
    images = [ex['image'] if 'image' in ex else None for ex in examples]

    # Filter out None images
    valid_samples = [(t, i) for t, i in zip(texts, images) if i is not None]
    if not valid_samples:
        return None

    texts, images = zip(*valid_samples)

    # Process
    batch = processor(
        text=list(texts),
        images=list(images),
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    batch["labels"] = batch["input_ids"].clone()

    return batch


def main():
    args = parse_args()

    # Load model and processor
    model, processor = load_model_processor(args)

    # Load dataset
    print(f"\nLoading dataset: {args.dataset}")
    if args.dataset_config:
        train_ds = load_dataset(args.dataset, args.dataset_config, split=args.split)
        eval_ds = load_dataset(args.dataset, args.dataset_config, split=args.eval_split)
    else:
        train_ds = load_dataset(args.dataset, split=args.split)
        eval_ds = load_dataset(args.dataset, split=args.eval_split)

    # Filter out samples without images (for ScienceQA)
    if 'image' in train_ds.column_names:
        original_size = len(train_ds)
        train_ds = train_ds.filter(lambda x: x['image'] is not None)
        print(f"Filtered training samples: {original_size} → {len(train_ds)} (with images)")

        original_size = len(eval_ds)
        eval_ds = eval_ds.filter(lambda x: x['image'] is not None)
        print(f"Filtered validation samples: {original_size} → {len(eval_ds)} (with images)")

    # Limit samples if specified
    if args.max_samples > 0:
        train_ds = train_ds.select(range(min(args.max_samples, len(train_ds))))
        print(f"Limited to {len(train_ds)} training samples")

    print(f"Final training samples: {len(train_ds)}")
    print(f"Final validation samples: {len(eval_ds)}")

    # Convert dataset to messages format for SFTTrainer
    def convert_to_vlm_format(example):
        """Convert ScienceQA to VLM training format"""
        # Format question with choices
        question = example['question']
        choices = example['choices']
        answer_idx = example['answer']

        choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
        prompt = f"Question: {question}\n\nChoices:\n{choices_text}\n\nAnswer:"

        # Answer
        answer_letter = chr(65 + answer_idx)
        answer_text = f" {answer_letter}. {choices[answer_idx]}"

        return {
            "image": example['image'],
            "text": prompt + answer_text,
        }

    # Apply conversion
    train_ds = train_ds.map(convert_to_vlm_format)
    eval_ds = eval_ds.map(convert_to_vlm_format)

    # Get LoRA config
    peft_cfg = get_lora_config(args)
    lr = args.lora_lr if args.method in ["lora-attention", "lora-all"] else args.fullft_lr

    # Apply LoRA if needed
    if peft_cfg is not None:
        model = get_peft_model(model, peft_cfg)
        model.print_trainable_parameters()

    # Training configuration
    training_config = {
        "output_dir": args.out,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": lr,
        "logging_steps": 10,
        "bf16": args.bf16 and not args.use_4bit,
        "fp16": not args.bf16,
        "max_seq_length": args.max_seq_length,
        "dataloader_num_workers": 4,
        "remove_unused_columns": False,  # Important for VLM!
        # Evaluation and checkpoint
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "save_total_limit": 3,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        # Gradient checkpointing (important for memory)
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
    }

    # Use max_steps if specified, otherwise use epochs
    if args.max_steps > 0:
        training_config["max_steps"] = args.max_steps
    else:
        training_config["num_train_epochs"] = args.num_train_epochs

    cfg = SFTConfig(**training_config)

    # Callbacks
    metrics_cb = MetricsCallback()
    early_stopping_cb = EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)

    # Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=processor.tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=cfg,
        callbacks=[metrics_cb, early_stopping_cb],
        data_collator=lambda examples: collate_fn(examples, processor),
    )

    print(f"\n{'='*60}")
    print(f"VLM Training Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Method: {args.method.upper()}")
    print(f"  Learning Rate: {lr}")
    print(f"  Epochs: {args.num_train_epochs if args.max_steps <= 0 else f'max_steps={args.max_steps}'}")
    print(f"  Early Stopping Patience: {args.early_stopping_patience} epochs")
    print(f"  Batch Size: {args.per_device_train_batch_size} x {args.grad_accum} = {args.per_device_train_batch_size * args.grad_accum}")
    print(f"  Max Sequence Length: {args.max_seq_length}")
    print(f"  Freeze Vision Tower: {args.freeze_vision_tower}")
    if args.method in ["lora-attention", "lora-all"]:
        print(f"  LoRA Rank: {args.lora_rank}")
        print(f"  LoRA Alpha: {args.lora_alpha}")
    print(f"{'='*60}\n")

    # Train
    trainer.train()

    # Save
    trainer.save_model()
    processor.save_pretrained(args.out)
    metrics_cb.save_metrics(args.out, args.method)

    print(f"\n✓ Training completed!")
    print(f"✓ Model saved to: {args.out}")


if __name__ == "__main__":
    main()


# Usage examples:

# LoRA-Attention on ScienceQA
# python3 sft_vlm_compare.py \
#   --model llava-hf/llava-1.5-7b-hf \
#   --dataset derek-thomas/ScienceQA \
#   --method lora-attention \
#   --out ./results/llava-scienceqa-lora-attention \
#   --bf16 --use_4bit \
#   --num_train_epochs 5 \
#   --per_device_train_batch_size 1 --grad_accum 4

# LoRA-All on ScienceQA (LoRA without Regret)
# python3 sft_vlm_compare.py \
#   --model llava-hf/llava-1.5-7b-hf \
#   --dataset derek-thomas/ScienceQA \
#   --method lora-all \
#   --out ./results/llava-scienceqa-lora-all \
#   --bf16 --use_4bit \
#   --num_train_epochs 5 \
#   --per_device_train_batch_size 1 --grad_accum 4

# Full Fine-Tuning on ScienceQA
# python3 sft_vlm_compare.py \
#   --model llava-hf/llava-1.5-7b-hf \
#   --dataset derek-thomas/ScienceQA \
#   --method fullft \
#   --out ./results/llava-scienceqa-fullft \
#   --bf16 --use_4bit \
#   --num_train_epochs 5 \
#   --per_device_train_batch_size 1 --grad_accum 4
