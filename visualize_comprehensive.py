#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GSM8K visualization comparing all experimental results"""
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def parse_experiment_name(name):
    """Parse experiment name into components"""
    parts = name.split('-')

    model = parts[0]  # llama or qwen
    dataset = parts[1]  # gsm8k

    # Method could be: lora-attention, lora-all, fullft
    if len(parts) >= 4 and parts[2] == "lora":
        method = f"lora-{parts[3]}"  # lora-attention or lora-all
    else:
        method = parts[2]  # fullft

    return {
        "model": model,
        "dataset": dataset,
        "method": method
    }


def load_results(results_dir):
    """Load all experimental results"""
    results_dir = Path(results_dir)

    experiments = {}

    for exp_dir in results_dir.glob("*"):
        if not exp_dir.is_dir():
            continue

        eval_file = exp_dir / "eval_results.json"

        # Parse the directory name to get the correct method
        info = parse_experiment_name(exp_dir.name)
        metrics_file = exp_dir / f"metrics_{info['method']}.json"

        if eval_file.exists() and metrics_file.exists():
            with open(eval_file) as f:
                eval_data = json.load(f)
            with open(metrics_file) as f:
                metrics_data = json.load(f)

            experiments[exp_dir.name] = {
                "eval": eval_data,
                "metrics": metrics_data
            }

    return experiments


def plot_comparison_grid(experiments, output_file):
    """Create GSM8K comparison grid"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('GSM8K Model Comparison: Llama-3.2-1B vs Qwen2.5-0.5B',
                 fontsize=16, fontweight='bold')

    # Parse experiment results (GSM8K only)
    results = {}
    for name, data in experiments.items():
        info = parse_experiment_name(name)

        # Skip non-GSM8K experiments
        if info['dataset'] != 'gsm8k':
            continue

        key = f"{info['model']}_{info['dataset']}_{info['method']}"

        # Extract metrics
        eval_data = data['eval']
        metrics = data['metrics']

        # Get final training metrics
        final_metrics = [m for m in metrics if m['loss'] is not None][-1]

        results[key] = {
            "model": info['model'],
            "dataset": info['dataset'],
            "method": info['method'],
            "eval_type": eval_data.get('eval_type', 'perplexity'),
            "performance": eval_data.get('accuracy', eval_data.get('perplexity')),
            "training_time": final_metrics['elapsed_time_min'],
            "gpu_memory": final_metrics['gpu_memory_gb'],
            "final_loss": final_metrics['loss'] if final_metrics['loss'] else 0,
            "metrics_history": metrics,  # Store full metrics history
        }

    # Plot 1: Performance Comparison
    ax = axes[0, 0]
    plot_performance(ax, results, "GSM8K Performance (Perplexity)")

    # Plot 2: Training Time Progression
    ax = axes[0, 1]
    plot_training_time(ax, results, "Training Time Progression")

    # Plot 3: GPU Memory Usage
    ax = axes[0, 2]
    plot_gpu_memory(ax, results, "GPU Memory Usage")

    # Plot 4: Training Loss Curves
    ax = axes[1, 0]
    plot_final_loss(ax, results, "Training Loss Curves")

    # Plot 5: Efficiency Score
    ax = axes[1, 1]
    plot_efficiency(ax, results, "Efficiency Score")

    # Plot 6: Method Comparison Summary
    ax = axes[1, 2]
    plot_method_summary(ax, results, "Method Comparison")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ GSM8K comparison saved to: {output_file}")


def plot_performance(ax, results, title):
    """Plot performance metrics"""
    if not results:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        ax.set_title(title)
        return

    models = ['llama', 'qwen']
    methods = ['lora-attention', 'lora-all', 'fullft']

    x = np.arange(len(models))
    width = 0.25

    lora_attn_vals = []
    lora_all_vals = []
    fullft_vals = []

    for model in models:
        lora_attn_key = f"{model}_gsm8k_lora-attention"
        lora_all_key = f"{model}_gsm8k_lora-all"
        fullft_key = f"{model}_gsm8k_fullft"

        lora_attn_vals.append(results.get(lora_attn_key, {}).get('performance', 0))
        lora_all_vals.append(results.get(lora_all_key, {}).get('performance', 0))
        fullft_vals.append(results.get(fullft_key, {}).get('performance', 0))

    ax.bar(x - width, lora_attn_vals, width, label='LoRA-Attn', alpha=0.8)
    ax.bar(x, lora_all_vals, width, label='LoRA-All', alpha=0.8)
    ax.bar(x + width, fullft_vals, width, label='Full FT', alpha=0.8)

    ax.set_xlabel('Model')
    ax.set_ylabel('Perplexity (lower is better)')
    ax.set_title(title, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in models])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)


def plot_training_time(ax, results, title):
    """Plot training time progression as line graph"""
    models = ['llama', 'qwen']
    methods = ['lora-attention', 'lora-all', 'fullft']
    method_labels = {'lora-attention': 'LoRA-Attn', 'lora-all': 'LoRA-All', 'fullft': 'Full FT'}
    colors = {'lora-attention': '#1f77b4', 'lora-all': '#ff7f0e', 'fullft': '#2ca02c'}
    linestyles = {'llama': '-', 'qwen': '--'}

    for method in methods:
        for model in models:
            key = f"{model}_gsm8k_{method}"
            result = results.get(key, {})
            metrics_history = result.get('metrics_history', [])

            if not metrics_history:
                continue

            # Extract time progression
            times = [m['elapsed_time_min'] for m in metrics_history if m['elapsed_time_min'] is not None]
            steps = [m['step'] for m in metrics_history if m['elapsed_time_min'] is not None]

            if times:
                label = f"{method_labels[method]} ({model.upper()})"
                ax.plot(steps, times, label=label, color=colors[method],
                       linestyle=linestyles[model], linewidth=2, marker='o',
                       markersize=3, alpha=0.8)

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Elapsed Time (min)')
    ax.set_title(title, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)


def plot_gpu_memory(ax, results, title):
    """Plot GPU memory usage"""
    models = ['llama', 'qwen']

    x = np.arange(len(models))
    width = 0.25

    lora_attn_mem = []
    lora_all_mem = []
    fullft_mem = []

    for model in models:
        lora_attn_key = f"{model}_gsm8k_lora-attention"
        lora_all_key = f"{model}_gsm8k_lora-all"
        fullft_key = f"{model}_gsm8k_fullft"

        lora_attn_mem.append(results.get(lora_attn_key, {}).get('gpu_memory', 0))
        lora_all_mem.append(results.get(lora_all_key, {}).get('gpu_memory', 0))
        fullft_mem.append(results.get(fullft_key, {}).get('gpu_memory', 0))

    ax.bar(x - width, lora_attn_mem, width, label='LoRA-Attn', alpha=0.8)
    ax.bar(x, lora_all_mem, width, label='LoRA-All', alpha=0.8)
    ax.bar(x + width, fullft_mem, width, label='Full FT', alpha=0.8)

    ax.set_ylabel('GPU Memory (GB)')
    ax.set_xlabel('Model')
    ax.set_title(title, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in models])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=16, color='r', linestyle='--', alpha=0.5, label='16GB Limit')


def plot_final_loss(ax, results, title):
    """Plot training loss progression as line graph"""
    models = ['llama', 'qwen']
    methods = ['lora-attention', 'lora-all', 'fullft']
    method_labels = {'lora-attention': 'LoRA-Attn', 'lora-all': 'LoRA-All', 'fullft': 'Full FT'}
    colors = {'lora-attention': '#1f77b4', 'lora-all': '#ff7f0e', 'fullft': '#2ca02c'}
    linestyles = {'llama': '-', 'qwen': '--'}

    for method in methods:
        for model in models:
            key = f"{model}_gsm8k_{method}"
            result = results.get(key, {})
            metrics_history = result.get('metrics_history', [])

            if not metrics_history:
                continue

            # Extract loss progression (filter out None values)
            losses = [m['loss'] for m in metrics_history if m['loss'] is not None]
            steps = [m['step'] for m in metrics_history if m['loss'] is not None]

            if losses:
                label = f"{method_labels[method]} ({model.upper()})"
                ax.plot(steps, losses, label=label, color=colors[method],
                       linestyle=linestyles[model], linewidth=2, marker='o',
                       markersize=3, alpha=0.8)

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Training Loss')
    ax.set_title(title, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)


def plot_efficiency(ax, results, title):
    """Plot efficiency score (performance per minute)"""
    models = ['llama', 'qwen']

    x = np.arange(len(models))
    width = 0.25

    lora_attn_eff = []
    lora_all_eff = []
    fullft_eff = []

    for model in models:
        lora_attn_key = f"{model}_gsm8k_lora-attention"
        lora_all_key = f"{model}_gsm8k_lora-all"
        fullft_key = f"{model}_gsm8k_fullft"

        lora_attn_data = results.get(lora_attn_key, {})
        lora_all_data = results.get(lora_all_key, {})
        fullft_data = results.get(fullft_key, {})

        # For perplexity: lower is better, so score = 1/perplexity / time
        lora_attn_score = (1 / max(lora_attn_data.get('performance', 1), 0.1)) / max(lora_attn_data.get('training_time', 1), 1)
        lora_all_score = (1 / max(lora_all_data.get('performance', 1), 0.1)) / max(lora_all_data.get('training_time', 1), 1)
        fullft_score = (1 / max(fullft_data.get('performance', 1), 0.1)) / max(fullft_data.get('training_time', 1), 1)

        lora_attn_eff.append(lora_attn_score * 100)
        lora_all_eff.append(lora_all_score * 100)
        fullft_eff.append(fullft_score * 100)

    ax.bar(x - width, lora_attn_eff, width, label='LoRA-Attn', alpha=0.8)
    ax.bar(x, lora_all_eff, width, label='LoRA-All', alpha=0.8)
    ax.bar(x + width, fullft_eff, width, label='Full FT', alpha=0.8)

    ax.set_ylabel('Efficiency Score (higher is better)')
    ax.set_xlabel('Model')
    ax.set_title(title, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in models])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)


def plot_method_summary(ax, results, title):
    """Plot method comparison summary table"""
    ax.axis('tight')
    ax.axis('off')

    # Prepare data
    methods = ['LoRA-Attn', 'LoRA-All', 'Full FT']
    metrics = ['Avg Time (min)', 'Avg Memory (GB)', 'Avg Perplexity']

    table_data = []
    for method_name, method_key in [('LoRA-Attn', 'lora-attention'),
                                      ('LoRA-All', 'lora-all'),
                                      ('Full FT', 'fullft')]:
        # Average across both models
        keys = [f"llama_gsm8k_{method_key}", f"qwen_gsm8k_{method_key}"]

        avg_time = np.mean([results.get(k, {}).get('training_time', 0) for k in keys])
        avg_mem = np.mean([results.get(k, {}).get('gpu_memory', 0) for k in keys])
        avg_perf = np.mean([results.get(k, {}).get('performance', 0) for k in keys])

        table_data.append([f"{avg_time:.1f}", f"{avg_mem:.1f}", f"{avg_perf:.2f}"])

    table = ax.table(cellText=table_data, rowLabels=methods, colLabels=metrics,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    ax.set_title(title, fontweight='bold', pad=20)


def main():
    parser = argparse.ArgumentParser(description='GSM8K experiment visualization')
    parser.add_argument('--results_dir', default='./results', help='Directory containing all results')
    parser.add_argument('--output', default='gsm8k_comparison.png', help='Output file')
    args = parser.parse_args()

    print("Loading GSM8K experimental results...")
    experiments = load_results(args.results_dir)

    # Filter only GSM8K experiments
    gsm8k_experiments = {k: v for k, v in experiments.items() if 'gsm8k' in k}

    if not gsm8k_experiments:
        print("Error: No GSM8K experimental results found!")
        return

    print(f"Found {len(gsm8k_experiments)} GSM8K experiments")
    for name in sorted(gsm8k_experiments.keys()):
        print(f"  - {name}")

    print("\nGenerating GSM8K comparison...")
    plot_comparison_grid(gsm8k_experiments, args.output)

    print("\nDone!")


if __name__ == "__main__":
    main()
