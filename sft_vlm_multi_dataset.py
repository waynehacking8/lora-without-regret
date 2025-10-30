#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA without Regret for VLMs - Multi-Dataset Support
Supports: ScienceQA, ChartQA, A-OKVQA
"""
import argparse
import json
import time
import torch
from datasets import load_dataset, Features, Sequence, Value, Image as ImageFeature
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    TrainerCallback,
    EarlyStoppingCallback,
    BitsAndBytesConfig
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model
import io
from PIL import Image


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
    p = argparse.ArgumentParser(description="VLM LoRA without Regret - Multi-Dataset")

    # Model arguments
    p.add_argument("--model", default="llava-hf/llava-1.5-7b-hf", help="VLM model name")
    p.add_argument("--method", choices=["lora-attention", "lora-all", "fullft"], required=True)

    # Dataset arguments
    p.add_argument("--dataset", required=True,
                   help="Dataset: scienceqa, chartqa, aokvqa")
    p.add_argument("--dataset_path", default=None, help="HuggingFace dataset path")
    p.add_argument("--split", default="train", help="Training dataset split")
    p.add_argument("--eval_split", default="validation[:200]", help="Validation split")
    p.add_argument("--max_samples", type=int, default=-1, help="Max training samples")

    # Output
    p.add_argument("--out", required=True, help="Output directory")

    # Training hyperparameters
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--early_stopping_patience", type=int, default=2)

    # LoRA configuration
    p.add_argument("--lora_rank", type=int, default=256)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_lr", type=float, default=2e-4)
    p.add_argument("--fullft_lr", type=float, default=2e-5)

    # Precision
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--use_4bit", action="store_true")

    # Vision encoder
    p.add_argument("--freeze_vision_tower", action="store_true", default=True)

    return p.parse_args()


def format_scienceqa(example):
    """Convert ScienceQA to messages format (multiple choice)"""
    question = example['question']
    choices = example['choices']
    answer_idx = example['answer']

    choices_text = "\n".join([f"{chr(65+i)}. {choice}"
                              for i, choice in enumerate(choices)])

    user_content = f"Question: {question}\n\nChoices:\n{choices_text}\n\nAnswer:"
    answer_letter = chr(65 + answer_idx)

    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": answer_letter}
    ]

    return {
        "messages": messages,
        "images": [example['image']]
    }


def format_chartqa(example):
    """Convert ChartQA to messages format (open-ended QA)"""
    query = example['query']
    label = str(example['label'])  # Convert to string

    # Convert bytes to PIL Image if needed
    if isinstance(example['image'], bytes):
        image = Image.open(io.BytesIO(example['image']))
    else:
        image = example['image']

    user_content = f"Question: {query}\n\nAnswer:"

    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": label}
    ]

    return {
        "messages": messages,
        "images": [image]
    }


def format_aokvqa(example):
    """Convert A-OKVQA to messages format (multiple choice)"""
    question = example['question']
    choices = example['choices']
    correct_idx = example['correct_choice_idx']

    choices_text = "\n".join([f"{chr(65+i)}. {choice}"
                              for i, choice in enumerate(choices)])

    user_content = f"Question: {question}\n\nChoices:\n{choices_text}\n\nAnswer:"
    answer_letter = chr(65 + correct_idx)

    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": answer_letter}
    ]

    return {
        "messages": messages,
        "images": [example['image']]
    }


def prepare_dataset(dataset, processor, dataset_name):
    """
    Prepare dataset in the format expected by SFTTrainer for VLMs
    """
    # Filter out samples without images
    dataset = dataset.filter(lambda x: x['image'] is not None,
                            desc="Filtering samples with images")

    # Select formatting function based on dataset
    if dataset_name == "scienceqa":
        format_fn = format_scienceqa
    elif dataset_name == "chartqa":
        format_fn = format_chartqa
    elif dataset_name == "aokvqa":
        format_fn = format_aokvqa
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # ✅ CRITICAL: Define features to preserve PIL Image format
    features = Features({
        'messages': [{'role': Value('string'), 'content': Value('string')}],
        'images': Sequence(ImageFeature(decode=True))
    })

    # Apply formatting
    formatted_dataset = dataset.map(
        format_fn,
        remove_columns=dataset.column_names,
        features=features,
        desc=f"Formatting {dataset_name}"
    )

    return formatted_dataset


def get_lora_config(args):
    """Configure LoRA based on method"""
    if args.method == "lora-attention":
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        print(f"LoRA-Attention: {len(target_modules)} modules")

    elif args.method == "lora-all":
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
            "gate_proj", "up_proj", "down_proj",      # MLP
        ]
        print(f"LoRA-All: {len(target_modules)} modules")

    else:
        return None

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )

    return lora_config


def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"LoRA without Regret for VLMs - {args.dataset.upper()}")
    print(f"Method: {args.method}")
    print(f"{'='*60}\n")

    # Load processor
    print(f"Loading processor: {args.model}")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

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
    print(f"Loading model: {args.model}")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if args.bf16 and not args.use_4bit else torch.float16,
        trust_remote_code=True,
    )

    # Freeze vision tower if requested
    if args.freeze_vision_tower and hasattr(model, 'vision_tower'):
        print("Freezing vision tower...")
        for param in model.vision_tower.parameters():
            param.requires_grad = False

    # Apply LoRA if needed
    if args.method in ["lora-attention", "lora-all"]:
        lora_config = get_lora_config(args)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Load dataset
    print(f"\nLoading dataset: {args.dataset}")

    # Map dataset name to HuggingFace path
    dataset_paths = {
        "scienceqa": "derek-thomas/ScienceQA",
        "chartqa": "ahmed-masry/ChartQA",
        "aokvqa": "HuggingFaceM4/A-OKVQA"
    }

    # Map dataset to correct validation split name
    validation_splits = {
        "scienceqa": "validation[:200]",
        "chartqa": "val[:200]",  # ChartQA uses 'val' not 'validation'
        "aokvqa": "validation[:200]"
    }

    dataset_path = args.dataset_path or dataset_paths.get(args.dataset)
    if not dataset_path:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Use dataset-specific validation split if not provided
    eval_split = args.eval_split
    if eval_split == "validation[:200]" and args.dataset in validation_splits:
        eval_split = validation_splits[args.dataset]

    train_ds = load_dataset(dataset_path, split=args.split)
    eval_ds = load_dataset(dataset_path, split=eval_split)

    print(f"Train samples: {len(train_ds)}")
    print(f"Eval samples: {len(eval_ds)}")

    # Prepare datasets
    train_ds = prepare_dataset(train_ds, processor, args.dataset)
    eval_ds = prepare_dataset(eval_ds, processor, args.dataset)

    print(f"After filtering:")
    print(f"  Train: {len(train_ds)} samples")
    print(f"  Eval: {len(eval_ds)} samples")

    # Training configuration
    learning_rate = args.lora_lr if args.method in ["lora-attention", "lora-all"] else args.fullft_lr

    training_config = {
        "output_dir": args.out,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps if args.max_steps > 0 else -1,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": learning_rate,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "logging_steps": 10,
        "eval_strategy": "steps",
        "eval_steps": 100,
        "save_strategy": "steps",
        "save_steps": 100,
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "max_length": None,  # ✅ CRITICAL for VLMs!
        "gradient_checkpointing": True,
        "remove_unused_columns": False,
        "report_to": "none",
        "bf16": args.bf16 and not args.use_4bit,
        "fp16": not args.bf16 or args.use_4bit,
        "dataloader_num_workers": 4,
        "dataloader_prefetch_factor": 2,
    }

    training_args = SFTConfig(**training_config)
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)

    # Initialize callbacks
    metrics_cb = MetricsCallback()
    early_stopping_cb = EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)

    # Create trainer
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=processor,  # ✅ Pass processor, not tokenizer!
        callbacks=[metrics_cb, early_stopping_cb],
    )

    # Train
    trainer.train()

    # Save model
    trainer.save_model(args.out)
    processor.save_pretrained(args.out)

    # Save metrics
    metrics_cb.save_metrics(args.out, args.method)

    print(f"\n{'='*60}")
    print(f"✅ Training completed!")
    print(f"Model saved to: {args.out}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
