#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
預先下載 VLM 模型和資料集
避免訓練時重複下載，節省時間
"""
import os
import argparse
from pathlib import Path
from huggingface_hub import login, snapshot_download
from datasets import load_dataset


def parse_args():
    p = argparse.ArgumentParser(description="預先下載 VLM 模型和資料集")
    p.add_argument("--token_file", default="./huggingface_token.txt",
                   help="HuggingFace token 文件路徑")
    p.add_argument("--cache_dir", default=None,
                   help="下載目錄 (預設使用 HuggingFace cache)")
    p.add_argument("--models", nargs="+",
                   default=["llava-hf/llava-1.5-7b-hf"],
                   help="要下載的模型列表")
    p.add_argument("--datasets", nargs="+",
                   default=["derek-thomas/ScienceQA"],
                   help="要下載的資料集列表")
    p.add_argument("--skip_models", action="store_true",
                   help="跳過模型下載")
    p.add_argument("--skip_datasets", action="store_true",
                   help="跳過資料集下載")
    return p.parse_args()


def load_token(token_file):
    """從文件讀取 HuggingFace token"""
    token_path = Path(token_file)
    if not token_path.exists():
        print(f"❌ Token 文件不存在: {token_file}")
        print("請創建文件並填入你的 HuggingFace token")
        return None

    with open(token_path, "r") as f:
        token = f.read().strip()

    if not token:
        print(f"❌ Token 文件為空: {token_file}")
        return None

    return token


def download_model(model_name, cache_dir=None):
    """下載 HuggingFace 模型"""
    print(f"\n{'='*60}")
    print(f"下載模型: {model_name}")
    print(f"{'='*60}")

    try:
        # Download entire model repository
        local_path = snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            resume_download=True,
            local_files_only=False,
        )
        print(f"✅ 模型下載完成!")
        print(f"   路徑: {local_path}")
        return True
    except Exception as e:
        print(f"❌ 下載失敗: {e}")
        return False


def download_dataset(dataset_name, cache_dir=None):
    """下載 HuggingFace 資料集"""
    print(f"\n{'='*60}")
    print(f"下載資料集: {dataset_name}")
    print(f"{'='*60}")

    try:
        # Download all splits
        print("正在下載 train split...")
        train_ds = load_dataset(
            dataset_name,
            split="train",
            cache_dir=cache_dir,
        )
        print(f"✓ Train: {len(train_ds)} samples")

        print("正在下載 validation split...")
        val_ds = load_dataset(
            dataset_name,
            split="validation",
            cache_dir=cache_dir,
        )
        print(f"✓ Validation: {len(val_ds)} samples")

        print("正在下載 test split...")
        test_ds = load_dataset(
            dataset_name,
            split="test",
            cache_dir=cache_dir,
        )
        print(f"✓ Test: {len(test_ds)} samples")

        print(f"✅ 資料集下載完成!")
        return True
    except Exception as e:
        print(f"❌ 下載失敗: {e}")
        return False


def main():
    args = parse_args()

    print("="*60)
    print("VLM 模型和資料集下載工具")
    print("="*60)

    # Load and login with token
    print(f"\n讀取 HuggingFace token: {args.token_file}")
    token = load_token(args.token_file)

    if token:
        print("登入 HuggingFace Hub...")
        try:
            login(token=token)
            print("✅ 登入成功!")
        except Exception as e:
            print(f"❌ 登入失敗: {e}")
            print("繼續嘗試下載...")
    else:
        print("⚠️  未提供 token，某些模型可能無法下載")

    # Summary
    total_downloads = 0
    successful_downloads = 0

    # Download models
    if not args.skip_models:
        print(f"\n{'='*60}")
        print(f"開始下載 {len(args.models)} 個模型")
        print(f"{'='*60}")

        for model_name in args.models:
            total_downloads += 1
            if download_model(model_name, args.cache_dir):
                successful_downloads += 1

    # Download datasets
    if not args.skip_datasets:
        print(f"\n{'='*60}")
        print(f"開始下載 {len(args.datasets)} 個資料集")
        print(f"{'='*60}")

        for dataset_name in args.datasets:
            total_downloads += 1
            if download_dataset(dataset_name, args.cache_dir):
                successful_downloads += 1

    # Final summary
    print(f"\n{'='*60}")
    print("下載完成!")
    print(f"{'='*60}")
    print(f"成功: {successful_downloads}/{total_downloads}")

    if successful_downloads == total_downloads:
        print("\n✅ 所有下載成功!")
        print("\n下一步:")
        print("  運行實驗: bash run_scienceqa_experiments.sh")
    else:
        print(f"\n⚠️  部分下載失敗 ({total_downloads - successful_downloads} 個)")
        print("請檢查錯誤訊息並重試")

    # Show cache location
    cache_dir = args.cache_dir or os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    print(f"\n📁 下載位置: {cache_dir}")


if __name__ == "__main__":
    main()


# Usage examples:

# 基本用法 (使用預設 token 文件)
# python3 download_models.py

# 指定 token 文件
# python3 download_models.py --token_file /path/to/token.txt

# 只下載模型
# python3 download_models.py --skip_datasets

# 只下載資料集
# python3 download_models.py --skip_models

# 下載多個模型
# python3 download_models.py --models llava-hf/llava-1.5-7b-hf llava-hf/llava-1.5-13b-hf

# 指定下載目錄
# python3 download_models.py --cache_dir /path/to/cache
