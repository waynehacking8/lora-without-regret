#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Professional visualization for VLM LoRA experiments"""
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


def parse_vlm_experiment_name(name):
    """Parse VLM experiment name (e.g., llava-scienceqa-lora-all-v2)"""
    parts = name.split('-')

    # llava-scienceqa-lora-all-v2
    model = parts[0]  # llava
    dataset = parts[1]  # scienceqa

    # Check for lora method
    if 'lora' in parts:
        lora_idx = parts.index('lora')
        if lora_idx + 1 < len(parts):
            # lora-all or lora-attention
            method = f"lora-{parts[lora_idx + 1]}"
        else:
            method = 'lora'
    elif 'fullft' in parts:
        method = 'fullft'
    else:
        method = 'unknown'

    return {"model": model, "dataset": dataset, "method": method}


def load_vlm_results(results_dir):
    """Load all VLM experimental results"""
    results_dir = Path(results_dir)
    experiments = {}

    for exp_dir in results_dir.glob("*"):
        if not exp_dir.is_dir():
            continue

        eval_file = exp_dir / "eval_results.json"
        info = parse_vlm_experiment_name(exp_dir.name)
        metrics_file = exp_dir / f"metrics_{info['method']}.json"

        if eval_file.exists() and metrics_file.exists():
            with open(eval_file) as f:
                eval_data = json.load(f)
            with open(metrics_file) as f:
                metrics_data = json.load(f)

            experiments[exp_dir.name] = {
                "info": info,
                "eval": eval_data,
                "metrics": metrics_data
            }

    return experiments


def plot_vlm_comparison(experiments, output_file, dataset_name=None):
    """Create professional VLM comparison visualization (2x2 grid)"""

    # Create 2x2 grid layout
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Auto-detect dataset
    if dataset_name is None:
        for name, data in experiments.items():
            dataset_name = data['info']['dataset']
            break

    # Dataset titles
    dataset_titles = {
        'scienceqa': 'ScienceQA Visual Reasoning',
        'aokvqa': 'A-OKVQA Commonsense VQA',
        'chartqa': 'ChartQA Mathematical Reasoning',
        'gqa': 'GQA Visual Reasoning'
    }

    title = dataset_titles.get(dataset_name, f'{dataset_name.upper()} VLM Benchmark')
    fig.suptitle(title + ' - Method Comparison', fontsize=16, fontweight='bold', y=0.98)

    # Group experiments by method
    methods_data = {}
    for name, data in experiments.items():
        method = data['info']['method']
        if method not in methods_data:
            methods_data[method] = []
        methods_data[method].append(data)

    # Sort methods
    method_order = ['lora-attention', 'lora-all', 'fullft']
    sorted_methods = [m for m in method_order if m in methods_data]

    # ========== Plot 1: Training Loss Progression (Top Left) ==========
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title('Training Loss Progression', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Training Steps', fontsize=10)
    ax1.set_ylabel('Training Loss', fontsize=10)
    ax1.grid(True, alpha=0.3)

    for method in sorted_methods:
        data_list = methods_data[method]
        for data in data_list:
            metrics = data['metrics']
            steps = [m['step'] for m in metrics if m.get('loss') is not None]
            losses = [m['loss'] for m in metrics if m.get('loss') is not None]

            if steps and losses:
                ax1.plot(steps, losses,
                        label=METHOD_LABELS[method],
                        color=COLORS[method],
                        linewidth=2,
                        alpha=0.9)

    ax1.legend(loc='upper right', framealpha=0.9)

    # ========== Plot 2: Final Accuracy (Top Right) ==========
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title('Final Accuracy (Higher is Better)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Method', fontsize=10)
    ax2.set_ylabel('Accuracy', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    accuracies = []
    labels = []
    colors = []

    for method in sorted_methods:
        data_list = methods_data[method]
        for data in data_list:
            eval_data = data['eval']
            acc = eval_data.get('accuracy', 0)

            accuracies.append(acc)
            labels.append(METHOD_LABELS[method])
            colors.append(COLORS[method])

    bars = ax2.bar(labels, accuracies, color=colors, edgecolor='black', linewidth=1.5, alpha=0.85)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                f'{acc:.1%}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax2.set_ylim(0, max(accuracies) * 1.15)

    # ========== Plot 3: Peak GPU Memory (Bottom Left) ==========
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_title('Peak GPU Memory Usage', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Method', fontsize=10)
    ax3.set_ylabel('GPU Memory (GB)', fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')

    # Add 96GB limit line
    ax3.axhline(y=96, color='red', linestyle='--', linewidth=2, label='96GB GPU Limit', alpha=0.7)

    gpu_mems = []
    labels_mem = []
    colors_mem = []

    for method in sorted_methods:
        data_list = methods_data[method]
        for data in data_list:
            metrics = data['metrics']
            max_gpu = max([m.get('gpu_memory_gb', 0) for m in metrics])

            gpu_mems.append(max_gpu)
            labels_mem.append(METHOD_LABELS[method])
            colors_mem.append(COLORS[method])

    bars = ax3.bar(labels_mem, gpu_mems, color=colors_mem, edgecolor='black', linewidth=1.5, alpha=0.85)

    # Add value labels
    for bar, mem in zip(bars, gpu_mems):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height,
                f'{mem:.1f} GB',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax3.legend(loc='upper right', framealpha=0.9)
    ax3.set_ylim(0, 100)

    # ========== Plot 4: Training Time (Bottom Right) ==========
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_title('Total Training Time', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Method', fontsize=10)
    ax4.set_ylabel('Training Time (minutes)', fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')

    times = []
    labels_time = []
    colors_time = []

    for method in sorted_methods:
        data_list = methods_data[method]
        for data in data_list:
            metrics = data['metrics']
            if metrics:
                total_time = metrics[-1].get('elapsed_time_min', 0)
                times.append(total_time)
                labels_time.append(METHOD_LABELS[method])
                colors_time.append(COLORS[method])

    bars = ax4.bar(labels_time, times, color=colors_time, edgecolor='black', linewidth=1.5, alpha=0.85)

    # Add value labels
    for bar, t in zip(bars, times):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2., height,
                f'{t:.1f} min',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax4.set_ylim(0, max(times) * 1.15 if times else 10)

    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Visualization saved to: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize VLM LoRA experiments")
    parser.add_argument("--results_dir", default="./results", help="Results directory")
    parser.add_argument("--output", default="vlm_comparison.png", help="Output file")
    parser.add_argument("--dataset", default=None, help="Dataset name (auto-detect if not specified)")
    args = parser.parse_args()

    print("Loading VLM experimental results...")
    experiments = load_vlm_results(args.results_dir)

    if not experiments:
        print(f"❌ No experiments found in {args.results_dir}")
        print("Expected directory structure:")
        print("  results/")
        print("    llava-scienceqa-lora-attention-v2/")
        print("      eval_results.json")
        print("      metrics_lora-attention.json")
        return

    print(f"Found {len(experiments)} experiments:")
    for name in experiments.keys():
        print(f"  - {name}")

    print(f"\nGenerating visualization...")
    plot_vlm_comparison(experiments, args.output, args.dataset)
    print(f"✓ Done!")


if __name__ == "__main__":
    main()


# Usage:
# python3 visualize_vlm.py --results_dir ./results --output scienceqa_professional.png
