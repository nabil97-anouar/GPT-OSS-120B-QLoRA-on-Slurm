#!/usr/bin/env python3
"""Merge a LoRA adapter into exportable 16-bit weights with Unsloth."""

from __future__ import annotations

import argparse
import os

import torch
from peft import PeftModel
from unsloth import FastLanguageModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-model",
        default=os.environ.get(
            "MODEL_ID", "unsloth/gpt-oss-120b-unsloth-bnb-4bit"
        ),
    )
    parser.add_argument(
        "--adapter",
        default=os.environ.get("OUT_DIR", "runs/gptoss120b-lora"),
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("MERGED_OUT", "runs/gptoss120b-merged"),
    )
    parser.add_argument("--max-seq-length", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter)

    if not hasattr(model, "save_pretrained_merged"):
        raise RuntimeError(
            "This Unsloth/PEFT combination does not expose save_pretrained_merged. "
            "Upgrade the environment or publish the adapter without merging."
        )

    model.save_pretrained_merged(
        args.output_dir,
        tokenizer,
        save_method="merged_16bit",
    )
    print(f"Merged model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
