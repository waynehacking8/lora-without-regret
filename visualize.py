#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate LoRA vs Full Fine-tuning comparison charts"""
import argparse
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lora_dir", required=True, help="LoRA model output directory")
    p.add_argument("--fullft_dir", required=True, help="Full FT model output directory")
    p.add_argument("--output", default="comparison.png", help="Output image filename")
    return p.parse_args()


def load_metrics(method_dir, method_name):
    """Load training metrics"""
    metrics_file = f"{method_dir}/metrics_{method_name}.json"
    eval_file = f"{method_dir}/eval_results.json"

    with open(metrics_file, "r") as f:
        metrics = json.load(f)

    eval_results = None
    try:
        with open(eval_file, "r") as f:
            eval_results = json.load(f)
    except FileNotFoundError:
        print(f"Warning: Evaluation results not found at {eval_file}")

    return metrics, eval_results


def plot_comparison(lora_metrics, fullft_metrics, lora_eval, fullft_eval, output_file):
    """Generate 2x2 comparison charts"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('LoRA without Regret vs Full Fine-tuning Comparison', fontsize=16, fontweight='bold')

    # Extract data
    lora_steps = [m['step'] for m in lora_metrics if m['loss'] is not None]
    lora_loss = [m['loss'] for m in lora_metrics if m['loss'] is not None]
    lora_gpu = [m['gpu_memory_gb'] for m in lora_metrics if m['loss'] is not None]
    lora_time = [m['elapsed_time_min'] for m in lora_metrics if m['loss'] is not None]

    fullft_steps = [m['step'] for m in fullft_metrics if m['loss'] is not None]
    fullft_loss = [m['loss'] for m in fullft_metrics if m['loss'] is not None]
    fullft_gpu = [m['gpu_memory_gb'] for m in fullft_metrics if m['loss'] is not None]
    fullft_time = [m['elapsed_time_min'] for m in fullft_metrics if m['loss'] is not None]

    # 1. Training loss comparison
    ax1 = axes[0, 0]
    ax1.plot(lora_steps, lora_loss, label='LoRA (r=256)', linewidth=2, marker='o', markersize=3)
    ax1.plot(fullft_steps, fullft_loss, label='Full Fine-tuning', linewidth=2, marker='s', markersize=3)
    ax1.set_xlabel('Training Steps', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Training Loss Comparison', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. GPU memory usage comparison
    ax2 = axes[0, 1]
    ax2.plot(lora_steps, lora_gpu, label='LoRA (bf16)', linewidth=2, color='green', marker='o', markersize=3)
    ax2.plot(fullft_steps, fullft_gpu, label='Full FT (bf16)', linewidth=2, color='orange', marker='s', markersize=3)
    ax2.set_xlabel('Training Steps', fontsize=11)
    ax2.set_ylabel('GPU Memory (GB)', fontsize=11)
    ax2.set_title('GPU Memory Usage', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Training time comparison
    ax3 = axes[1, 0]
    if lora_time and fullft_time:
        ax3.plot(lora_steps, lora_time, label='LoRA', linewidth=2, color='blue', marker='o', markersize=3)
        ax3.plot(fullft_steps, fullft_time, label='Full FT', linewidth=2, color='red', marker='s', markersize=3)
        ax3.set_xlabel('Training Steps', fontsize=11)
        ax3.set_ylabel('Elapsed Time (minutes)', fontsize=11)
        ax3.set_title('Training Time Comparison', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Add speedup annotation
        final_lora_time = lora_time[-1] if lora_time else 0
        final_fullft_time = fullft_time[-1] if fullft_time else 0
        if final_fullft_time > 0:
            speedup = final_fullft_time / final_lora_time
            ax3.text(0.05, 0.95, f'LoRA speedup: {speedup:.2f}x',
                    transform=ax3.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 4. Perplexity comparison (if evaluation results exist)
    ax4 = axes[1, 1]
    if lora_eval and fullft_eval:
        methods = ['LoRA\n(r=256)', 'Full FT']
        perplexities = [lora_eval['perplexity'], fullft_eval['perplexity']]
        colors = ['skyblue', 'lightcoral']

        bars = ax4.bar(methods, perplexities, color=colors, edgecolor='black', linewidth=1.5)
        ax4.set_ylabel('Perplexity (lower is better)', fontsize=11)
        ax4.set_title('Test Perplexity Comparison', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')

        # Add values on bars
        for bar, ppl in zip(bars, perplexities):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{ppl:.2f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Calculate performance gap
        ppl_diff = abs(lora_eval['perplexity'] - fullft_eval['perplexity'])
        ppl_pct = (ppl_diff / fullft_eval['perplexity']) * 100
        ax4.text(0.5, 0.95, f'Gap: {ppl_diff:.2f} ({ppl_pct:.1f}%)',
                transform=ax4.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    else:
        ax4.text(0.5, 0.5, 'Evaluation results not found\nPlease run evaluate.py first',
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.axis('off')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison chart saved to: {output_file}")


def print_summary(lora_metrics, fullft_metrics, lora_eval, fullft_eval):
    """Print comparison summary"""
    print("\n" + "="*60)
    print("LoRA without Regret vs Full Fine-tuning Summary")
    print("="*60)

    # Training time
    lora_time = lora_metrics[-1]['elapsed_time_min'] if lora_metrics else 0
    fullft_time = fullft_metrics[-1]['elapsed_time_min'] if fullft_metrics else 0
    print(f"\nTraining Time:")
    print(f"  LoRA: {lora_time:.2f} minutes")
    print(f"  Full FT: {fullft_time:.2f} minutes")
    if fullft_time > 0:
        print(f"  Speedup: {fullft_time/lora_time:.2f}x")

    # GPU memory
    lora_gpu_max = max([m['gpu_memory_gb'] for m in lora_metrics if m['gpu_memory_gb']])
    fullft_gpu_max = max([m['gpu_memory_gb'] for m in fullft_metrics if m['gpu_memory_gb']])
    print(f"\nPeak GPU Memory:")
    print(f"  LoRA: {lora_gpu_max:.2f} GB")
    print(f"  Full FT: {fullft_gpu_max:.2f} GB")
    print(f"  Saved: {fullft_gpu_max - lora_gpu_max:.2f} GB ({(1-lora_gpu_max/fullft_gpu_max)*100:.1f}%)")

    # Final loss
    lora_final_loss = [m['loss'] for m in lora_metrics if m['loss']][-1]
    fullft_final_loss = [m['loss'] for m in fullft_metrics if m['loss']][-1]
    print(f"\nFinal Training Loss:")
    print(f"  LoRA: {lora_final_loss:.4f}")
    print(f"  Full FT: {fullft_final_loss:.4f}")

    # Perplexity
    if lora_eval and fullft_eval:
        print(f"\nTest Perplexity:")
        print(f"  LoRA: {lora_eval['perplexity']:.4f}")
        print(f"  Full FT: {fullft_eval['perplexity']:.4f}")
        diff_pct = abs(lora_eval['perplexity'] - fullft_eval['perplexity']) / fullft_eval['perplexity'] * 100
        print(f"  Gap: {diff_pct:.2f}%")

    print("="*60 + "\n")


def main():
    args = parse_args()

    print("Loading training metrics...")
    lora_metrics, lora_eval = load_metrics(args.lora_dir, "lora")
    fullft_metrics, fullft_eval = load_metrics(args.fullft_dir, "fullft")

    print("Generating comparison charts...")
    plot_comparison(lora_metrics, fullft_metrics, lora_eval, fullft_eval, args.output)

    print_summary(lora_metrics, fullft_metrics, lora_eval, fullft_eval)


if __name__ == "__main__":
    main()


# Usage example:
# python3 visualize.py --lora_dir ./lora-sft --fullft_dir ./fullft-sft --output comparison.png
