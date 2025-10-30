#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA without Regret for Vision-Language Models (VLMs) - CORRECTED VERSION
Based on official TRL examples: trl/examples/scripts/sft_vlm.py
"""
import argparse
import json
import time
import torch
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,  # ✅ Correct class for VLMs
    TrainerCallback,
    EarlyStoppingCallback,
    BitsAndBytesConfig
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model


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
    p = argparse.ArgumentParser(description="VLM LoRA without Regret experiments (CORRECTED)")

    # Model arguments
    p.add_argument("--model", default="llava-hf/llava-1.5-7b-hf", help="VLM model name")
    p.add_argument("--method", choices=["lora-attention", "lora-all", "fullft"], required=True)

    # Dataset arguments
    p.add_argument("--dataset", default="derek-thomas/ScienceQA", help="Dataset name")
    p.add_argument("--dataset_config", default=None, help="Dataset config (if applicable)")
    p.add_argument("--split", default="train", help="Training dataset split")
    p.add_argument("--eval_split", default="validation[:200]", help="Validation split")
    p.add_argument("--max_samples", type=int, default=-1, help="Max training samples (-1 for all)")

    # Output
    p.add_argument("--out", default="./out-vlm-sft", help="Output directory")

    # Training hyperparameters
    p.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    p.add_argument("--max_steps", type=int, default=-1, help="Max steps (overrides epochs if > 0)")
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps")
    p.add_argument("--early_stopping_patience", type=int, default=2, help="Early stopping patience")

    # LoRA configuration (LoRA without Regret)
    p.add_argument("--lora_rank", type=int, default=256, help="LoRA rank (default: 256, full LoRA without Regret)")
    p.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha (default: 16)")
    p.add_argument("--lora_lr", type=float, default=2e-4, help="LoRA learning rate")
    p.add_argument("--fullft_lr", type=float, default=2e-5, help="Full FT learning rate")

    # Precision
    p.add_argument("--bf16", action="store_true", help="Use bfloat16 precision")
    p.add_argument("--use_4bit", action="store_true", help="Use 4-bit quantization")

    # Vision encoder
    p.add_argument("--freeze_vision_tower", action="store_true", default=True,
                   help="Freeze vision encoder (recommended)")

    return p.parse_args()


def prepare_dataset(dataset, processor, dataset_name):
    """
    Prepare dataset in the format expected by SFTTrainer for VLMs

    Required format:
    - 'images' column with PIL images (note: plural!)
    - 'messages' column with conversation format
    """
    from datasets import Features, Sequence, Value, Image as ImageFeature

    # First filter out samples without images
    dataset = dataset.filter(lambda x: x['image'] is not None, desc="Filtering samples with images")

    def format_scienceqa(example):
        """Convert ScienceQA to messages format"""
        # Format question with choices
        question = example['question']
        choices = example['choices']
        answer_idx = example['answer']

        choices_text = "\n".join([f"{chr(65+i)}. {choice}"
                                  for i, choice in enumerate(choices)])

        # Create user message (question + choices)
        user_content = f"Question: {question}\n\nChoices:\n{choices_text}\n\nAnswer:"

        # Create assistant message (answer)
        answer_letter = chr(65 + answer_idx)
        assistant_content = f"{answer_letter}"

        # Format as messages
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]

        return {
            "messages": messages,
            "images": [example['image']]  # Note: list of PIL images
        }

    # ✅ CRITICAL: Define features to preserve PIL Image format
    features = Features({
        'messages': [{'role': Value('string'), 'content': Value('string')}],
        'images': Sequence(ImageFeature(decode=True))  # Keep as PIL Images!
    })

    # Apply formatting with features to preserve image format
    formatted_dataset = dataset.map(
        format_scienceqa,
        remove_columns=dataset.column_names,
        features=features,  # ✅ Preserve PIL Images!
        desc="Formatting dataset"
    )

    return formatted_dataset


def get_lora_config(args):
    """Configure LoRA based on method"""

    if args.method == "lora-attention":
        # LoRA only on language model attention layers
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        print(f"LoRA-Attention: {len(target_modules)} modules")

    elif args.method == "lora-all":
        # LoRA without Regret: all linear layers in language model
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
            "gate_proj", "up_proj", "down_proj",      # MLP
        ]
        print(f"LoRA-All: {len(target_modules)} modules")

    else:
        return None  # Full fine-tuning

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
    print("VLM LoRA without Regret - CORRECTED VERSION")
    print(f"{'='*60}\n")

    # Configure quantization
    quantization_config = None
    if args.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        print("✓ Using 4-bit quantization")

    # ✅ Load model with correct class
    print(f"Loading model: {args.model}")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if args.bf16 and not args.use_4bit else torch.float16,
        trust_remote_code=True,
    )

    # Load processor
    processor = AutoProcessor.from_pretrained(args.model)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # Freeze vision tower if specified
    if args.freeze_vision_tower and hasattr(model, 'vision_tower'):
        for param in model.vision_tower.parameters():
            param.requires_grad = False
        print("✓ Vision tower frozen")

    # Load datasets
    print(f"\nLoading dataset: {args.dataset}")
    if args.dataset_config:
        train_ds = load_dataset(args.dataset, args.dataset_config, split=args.split)
        eval_ds = load_dataset(args.dataset, args.dataset_config, split=args.eval_split)
    else:
        train_ds = load_dataset(args.dataset, split=args.split)
        eval_ds = load_dataset(args.dataset, split=args.eval_split)

    print(f"Raw training samples: {len(train_ds)}")
    print(f"Raw validation samples: {len(eval_ds)}")

    # Prepare datasets
    print("\nPreparing datasets...")
    train_ds = prepare_dataset(train_ds, processor, args.dataset)
    eval_ds = prepare_dataset(eval_ds, processor, args.dataset)

    print(f"Filtered training samples: {len(train_ds)}")
    print(f"Filtered validation samples: {len(eval_ds)}")

    # Limit samples if specified
    if args.max_samples > 0:
        train_ds = train_ds.select(range(min(args.max_samples, len(train_ds))))
        print(f"Limited to {len(train_ds)} training samples")

    # Get LoRA config and learning rate
    peft_config = get_lora_config(args)
    lr = args.lora_lr if args.method in ["lora-attention", "lora-all"] else args.fullft_lr

    # Apply LoRA if needed
    if peft_config is not None:
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # ✅ Correct training configuration
    training_config = {
        "output_dir": args.out,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": lr,
        "logging_steps": 10,
        "bf16": args.bf16 and not args.use_4bit,
        "fp16": not args.bf16 or args.use_4bit,
        # ✅ CRITICAL: max_length must be None for VLMs!
        "max_length": None,
        "dataset_text_field": "",  # Not used for VLMs with messages format
        "dataset_kwargs": {"skip_prepare_dataset": False},
        # Evaluation and checkpoint
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        # Gradient checkpointing
        "gradient_checkpointing": True,
        "remove_unused_columns": False,  # Important for VLMs!
        # Disable wandb
        "report_to": "none",
    }

    # Use max_steps if specified, otherwise use epochs
    if args.max_steps > 0:
        training_config["max_steps"] = args.max_steps
    else:
        training_config["num_train_epochs"] = args.num_train_epochs

    training_args = SFTConfig(**training_config)

    # ✅ CRITICAL: Set gradient checkpointing kwargs
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)

    # Callbacks
    metrics_cb = MetricsCallback()
    early_stopping_cb = EarlyStoppingCallback(
        early_stopping_patience=args.early_stopping_patience
    )

    # ✅ CRITICAL FIX: Pass processor (ProcessorMixin), not tokenizer!
    # TRL checks: isinstance(processing_class, ProcessorMixin) to detect VLMs
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=processor,  # ✅ Pass processor, not processor.tokenizer!
        callbacks=[metrics_cb, early_stopping_cb],
    )

    # Print configuration
    print(f"\n{'='*60}")
    print(f"Training Configuration:")
    print(f"  Method: {args.method.upper()}")
    print(f"  Learning Rate: {lr}")
    print(f"  Epochs: {args.num_train_epochs if args.max_steps <= 0 else f'max_steps={args.max_steps}'}")
    print(f"  Batch Size: {args.per_device_train_batch_size} x {args.grad_accum} = {args.per_device_train_batch_size * args.grad_accum}")
    print(f"  Max Length: {training_args.max_length} (None = no truncation)")
    print(f"  Freeze Vision: {args.freeze_vision_tower}")
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
# python3 sft_vlm_compare_v2.py \
#   --model llava-hf/llava-1.5-7b-hf \
#   --dataset derek-thomas/ScienceQA \
#   --method lora-attention \
#   --out ./results/llava-scienceqa-lora-attention-v2 \
#   --bf16 --use_4bit \
#   --num_train_epochs 3

# LoRA-All (LoRA without Regret)
# python3 sft_vlm_compare_v2.py \
#   --model llava-hf/llava-1.5-7b-hf \
#   --dataset derek-thomas/ScienceQA \
#   --method lora-all \
#   --out ./results/llava-scienceqa-lora-all-v2 \
#   --bf16 --use_4bit \
#   --num_train_epochs 3

# Full Fine-Tuning
# python3 sft_vlm_compare_v2.py \
#   --model llava-hf/llava-1.5-7b-hf \
#   --dataset derek-thomas/ScienceQA \
#   --method fullft \
#   --out ./results/llava-scienceqa-fullft-v2 \
#   --bf16 --use_4bit \
#   --num_train_epochs 3
