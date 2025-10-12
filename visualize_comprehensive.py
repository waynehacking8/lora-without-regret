#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Professional visualization for LoRA experiments"""
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set professional style
plt.style.use('seaborn-v0_8-paper')
COLORS = {
    'lora-attention': '#2E86AB',  # Professional blue
    'lora-all': '#A23B72',        # Purple
    'fullft': '#F18F01'           # Orange
}

METHOD_LABELS = {
    'lora-attention': 'LoRA-Attention',
    'lora-all': 'LoRA-All',
    'fullft': 'Full Fine-Tuning'
}


def parse_experiment_name(name):
    """Parse experiment name into components"""
    parts = name.split('-')

    model = parts[0]

    # Handle compound dataset names (e.g., arc-easy, arc-challenge)
    if len(parts) >= 3 and parts[2] in ['easy', 'challenge']:
        dataset = f"{parts[1]}-{parts[2]}"
        # Check if method is lora (e.g., llama-arc-easy-lora-attention)
        if len(parts) >= 5 and parts[3] == "lora":
            method = f"lora-{parts[4]}"
        else:
            # Full FT case (e.g., llama-arc-easy-fullft)
            method = parts[3]
    else:
        # Simple dataset names (e.g., gsm8k, csqa)
        dataset = parts[1]
        if len(parts) >= 4 and parts[2] == "lora":
            method = f"lora-{parts[3]}"
        else:
            method = parts[2]

    return {"model": model, "dataset": dataset, "method": method}


def load_results(results_dir):
    """Load all experimental results"""
    results_dir = Path(results_dir)
    experiments = {}

    for exp_dir in results_dir.glob("*"):
        if not exp_dir.is_dir():
            continue

        eval_file = exp_dir / "eval_results.json"
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
    """Create professional comparison visualization"""

    # Create 2x2 grid layout
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Determine dataset type
    dataset_type = None
    for name in experiments.keys():
        info = parse_experiment_name(name)
        dataset_type = info['dataset']
        break

    # Set title
    dataset_names = {
        'gsm8k': 'GSM8K Mathematical Reasoning',
        'csqa': 'CommonsenseQA',
        'arc': 'ARC Reasoning Challenge',
        'arc-easy': 'ARC-Easy Reasoning Challenge',
        'arc-challenge': 'ARC-Challenge Reasoning Challenge',
        'mmlu': 'MMLU Benchmark'
    }
    fig.suptitle(f'{dataset_names.get(dataset_type, dataset_type.upper())} - Method Comparison',
                 fontsize=18, fontweight='bold', y=0.98)

    # Parse results
    results = {}
    for name, data in experiments.items():
        info = parse_experiment_name(name)
        key = f"{info['model']}_{info['dataset']}_{info['method']}"

        eval_data = data['eval']
        metrics = data['metrics']
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
            "metrics_history": metrics,
        }

    # Create subplots
    ax1 = fig.add_subplot(gs[0, 0])  # Training Loss
    plot_training_loss(ax1, results, dataset_type)

    ax2 = fig.add_subplot(gs[0, 1])  # Performance
    plot_performance(ax2, results, dataset_type)

    ax3 = fig.add_subplot(gs[1, 0])  # GPU Memory
    plot_gpu_memory(ax3, results, dataset_type)

    ax4 = fig.add_subplot(gs[1, 1])  # Training Time
    plot_training_time(ax4, results, dataset_type)

    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Visualization saved to: {output_file}")


def plot_training_loss(ax, results, dataset_type):
    """Plot training loss curves"""
    methods = ['lora-attention', 'lora-all', 'fullft']

    for method in methods:
        key = f"llama_{dataset_type}_{method}"
        result = results.get(key, {})
        metrics_history = result.get('metrics_history', [])

        if not metrics_history:
            continue

        losses = [m['loss'] for m in metrics_history if m['loss'] is not None]
        steps = [m['step'] for m in metrics_history if m['loss'] is not None]

        if losses:
            ax.plot(steps, losses, label=METHOD_LABELS[method],
                   color=COLORS[method], linewidth=2.5, alpha=0.9)

    ax.set_xlabel('Training Steps', fontsize=11, fontweight='500')
    ax.set_ylabel('Training Loss', fontsize=11, fontweight='500')
    ax.set_title('Training Loss Progression', fontsize=13, fontweight='600', pad=10)
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_performance(ax, results, dataset_type):
    """Plot final performance comparison"""
    methods = ['lora-attention', 'lora-all', 'fullft']
    method_labels = [METHOD_LABELS[m] for m in methods]

    vals = []
    colors = []
    for method in methods:
        key = f"llama_{dataset_type}_{method}"
        vals.append(results.get(key, {}).get('performance', 0))
        colors.append(COLORS[method])

    x = np.arange(len(methods))
    bars = ax.bar(x, vals, color=colors, alpha=0.85, edgecolor='black', linewidth=1.2)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10, fontweight='600')

    ax.set_xlabel('Method', fontsize=11, fontweight='500')

    if dataset_type in ['gsm8k', 'mmlu']:
        ylabel = 'Perplexity ↓'
        title = 'Final Perplexity (Lower is Better)'
    else:
        ylabel = 'Accuracy ↑'
        title = 'Test Accuracy (Higher is Better)'

    ax.set_ylabel(ylabel, fontsize=11, fontweight='500')
    ax.set_title(title, fontsize=13, fontweight='600', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.25, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_gpu_memory(ax, results, dataset_type):
    """Plot GPU memory usage comparison"""
    methods = ['lora-attention', 'lora-all', 'fullft']
    method_labels = [METHOD_LABELS[m] for m in methods]

    mem_vals = []
    colors = []
    for method in methods:
        key = f"llama_{dataset_type}_{method}"
        mem_vals.append(results.get(key, {}).get('gpu_memory', 0))
        colors.append(COLORS[method])

    x = np.arange(len(methods))
    bars = ax.bar(x, mem_vals, color=colors, alpha=0.85, edgecolor='black', linewidth=1.2)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f} GB',
                   ha='center', va='bottom', fontsize=10, fontweight='600')

    # Add 16GB reference line
    ax.axhline(y=16, color='red', linestyle='--', linewidth=2,
               alpha=0.7, label='16GB GPU Limit')

    ax.set_xlabel('Method', fontsize=11, fontweight='500')
    ax.set_ylabel('GPU Memory (GB)', fontsize=11, fontweight='500')
    ax.set_title('Peak GPU Memory Usage', fontsize=13, fontweight='600', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, rotation=15, ha='right')
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=9)
    ax.grid(axis='y', alpha=0.25, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_training_time(ax, results, dataset_type):
    """Plot training time comparison"""
    methods = ['lora-attention', 'lora-all', 'fullft']
    method_labels = [METHOD_LABELS[m] for m in methods]

    time_vals = []
    colors = []
    for method in methods:
        key = f"llama_{dataset_type}_{method}"
        time_vals.append(results.get(key, {}).get('training_time', 0))
        colors.append(COLORS[method])

    x = np.arange(len(methods))
    bars = ax.bar(x, time_vals, color=colors, alpha=0.85, edgecolor='black', linewidth=1.2)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f} min',
                   ha='center', va='bottom', fontsize=10, fontweight='600')

    ax.set_xlabel('Method', fontsize=11, fontweight='500')
    ax.set_ylabel('Training Time (minutes)', fontsize=11, fontweight='500')
    ax.set_title('Total Training Time', fontsize=13, fontweight='600', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.25, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def main():
    parser = argparse.ArgumentParser(description='Professional experiment visualization')
    parser.add_argument('--results_dir', default='./results', help='Directory containing results')
    parser.add_argument('--output', default='comparison.png', help='Output file')
    args = parser.parse_args()

    print("Loading experimental results...")
    experiments = load_results(args.results_dir)

    if not experiments:
        print("Error: No experimental results found!")
        return

    print(f"Found {len(experiments)} experiments")
    for name in sorted(experiments.keys()):
        print(f"  - {name}")

    print("\nGenerating professional visualization...")
    plot_comparison_grid(experiments, args.output)

    print("\nDone!")


if __name__ == "__main__":
    main()
