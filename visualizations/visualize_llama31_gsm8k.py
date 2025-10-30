#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate professional visualization for Llama-3.1-8B GSM8K experiments
Same style as ScienceQA visualization
"""
import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Set style (same as ScienceQA)
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

# Define colors (same as ScienceQA)
COLORS = {
    'lora-attention': '#5B9BD5',  # Blue
    'lora-all': '#C55A9E',         # Purple/Magenta
    'fullft': '#ED7D31'            # Orange
}

METHODS = ['lora-attention', 'lora-all', 'fullft']
METHOD_NAMES = {
    'lora-attention': 'LoRA-Attention',
    'lora-all': 'LocalMind',
    'fullft': 'Full Fine-Tuning'
}

def load_metrics(method):
    """Load training metrics"""
    path = f'./results/llama31-gsm8k-{method}/metrics_{method}.json'
    with open(path) as f:
        return json.load(f)

def load_eval_results(method):
    """Load evaluation results"""
    path = f'./results/llama31-gsm8k-{method}/eval_results.json'
    with open(path) as f:
        return json.load(f)

def plot_llama31_gsm8k_comparison():
    """Generate comprehensive comparison plot"""

    # Load all data
    all_metrics = {}
    all_eval = {}

    for method in METHODS:
        all_metrics[method] = load_metrics(method)
        all_eval[method] = load_eval_results(method)

    # Create figure with 2x2 grid
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # ============================================================
    # 1. Top Left: Training Loss Progression
    # ============================================================
    ax1 = fig.add_subplot(gs[0, 0])

    for method in METHODS:
        metrics = all_metrics[method]
        steps = [m['step'] for m in metrics if m.get('loss')]
        losses = [m['loss'] for m in metrics if m.get('loss')]

        ax1.plot(steps, losses,
                label=METHOD_NAMES[method],
                color=COLORS[method],
                linewidth=2,
                alpha=0.8)

    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss Progression', fontweight='bold', pad=15)
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)

    # ============================================================
    # 2. Top Right: Final Perplexity
    # ============================================================
    ax2 = fig.add_subplot(gs[0, 1])

    perplexities = []
    labels = []
    colors_list = []

    for method in METHODS:
        eval_results = all_eval[method]
        perplexity = eval_results['perplexity']
        perplexities.append(perplexity)
        labels.append(METHOD_NAMES[method])
        colors_list.append(COLORS[method])

    bars = ax2.bar(labels, perplexities, color=colors_list,
                   edgecolor='black', linewidth=1.5, alpha=0.8)

    # Add value labels on bars
    for bar, ppl in zip(bars, perplexities):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{ppl:.2f}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax2.set_ylabel('Perplexity')
    ax2.set_title('Final Perplexity (Lower is Better)', fontweight='bold', pad=15)
    ax2.set_ylim(0, max(perplexities) * 1.15)
    ax2.grid(True, alpha=0.3, axis='y')

    # ============================================================
    # 3. Bottom Left: Peak GPU Memory Usage
    # ============================================================
    ax3 = fig.add_subplot(gs[1, 0])

    peak_mems = []
    labels = []
    colors_list = []

    for method in METHODS:
        metrics = all_metrics[method]
        peak_mem = max([m['gpu_memory_gb'] for m in metrics if m.get('gpu_memory_gb')])
        peak_mems.append(peak_mem)
        labels.append(METHOD_NAMES[method])
        colors_list.append(COLORS[method])

    bars = ax3.bar(labels, peak_mems, color=colors_list,
                   edgecolor='black', linewidth=1.5, alpha=0.8)

    # Add value labels on bars
    for bar, mem in zip(bars, peak_mems):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{mem:.1f} GB',
                ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Add 96GB limit line
    ax3.axhline(y=96, color='red', linestyle='--', linewidth=2,
                label='96GB GPU Limit', alpha=0.7)

    ax3.set_ylabel('GPU Memory (GB)')
    ax3.set_title('Peak GPU Memory Usage', fontweight='bold', pad=15)
    ax3.set_ylim(0, 100)
    ax3.legend(frameon=True, fancybox=True, shadow=True)
    ax3.grid(True, alpha=0.3, axis='y')

    # ============================================================
    # 4. Bottom Right: Total Training Time
    # ============================================================
    ax4 = fig.add_subplot(gs[1, 1])

    training_times = []
    labels = []
    colors_list = []

    for method in METHODS:
        metrics = all_metrics[method]
        total_time_min = metrics[-1]['elapsed_time_min']
        training_times.append(total_time_min)
        labels.append(METHOD_NAMES[method])
        colors_list.append(COLORS[method])

    bars = ax4.bar(labels, training_times, color=colors_list,
                   edgecolor='black', linewidth=1.5, alpha=0.8)

    # Add value labels on bars
    for bar, time in zip(bars, training_times):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.1f} min',
                ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax4.set_ylabel('Training Time (minutes)')
    ax4.set_title('Total Training Time', fontweight='bold', pad=15)
    ax4.set_ylim(0, max(training_times) * 1.15)
    ax4.grid(True, alpha=0.3, axis='y')

    # ============================================================
    # Main title
    # ============================================================
    fig.suptitle('Llama-3.1-8B-Instruct GSM8K - Method Comparison',
                 fontsize=18, fontweight='bold', y=0.98)

    # Save figure
    os.makedirs('./visualizations', exist_ok=True)
    output_path = './visualizations/llama31_gsm8k_professional.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'✓ Saved visualization to: {output_path}')

    return output_path

if __name__ == '__main__':
    print('='*60)
    print('Generating Llama-3.1-8B GSM8K Visualization')
    print('='*60)
    print()

    output_path = plot_llama31_gsm8k_comparison()

    print()
    print('='*60)
    print('✅ Visualization completed!')
    print('='*60)
