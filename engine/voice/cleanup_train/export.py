"""Exports a trained clean-up adapter for llama.cpp: merged into Qwen3.5-4B, converted to GGUF, quantized.

    .\\.venv\\Scripts\\python.exe -m engine.voice.cleanup_train.export --run full1
    .\\.venv\\Scripts\\python.exe -m engine.voice.cleanup_train.export --run full1 --adapter epoch2 --quant Q8_0

Writes checkpoints/cleanup/<run>/merged/ (the base model with the adapter merged in, Hugging Face format, about
8 GB) and checkpoints/cleanup/<run>/gguf/: <label>-bf16.gguf and one file per quantization (Q8_0 and Q4_K_M by
default). checkpoints/ is git-ignored.

Needs llama.cpp in downloads/llama.cpp/b11960/ (git-ignored; MIT licence, see docs/DECISIONS.md): bin/ holds the
Windows CUDA 13.4 build with its CUDA runtime DLLs, src/ the source of the same commit (v0.8.0), whose converter
reads Qwen3.5. The merge runs on the CPU, as the eval's merge does, so the GPU stays free.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CHECKPOINTS = ROOT / "checkpoints" / "cleanup"
LLAMA_CPP = ROOT / "downloads" / "llama.cpp" / "b11960"
CONVERTER = LLAMA_CPP / "src" / "llama.cpp-0.8.0" / "convert_hf_to_gguf.py"
QUANTIZE = LLAMA_CPP / "bin" / "llama-quantize.exe"


def label(run: str, adapter: str = "adapter") -> str:
    """full1 for the final adapter, full1-epoch2 for an earlier epoch (as the eval names its runs)."""
    return run if adapter == "adapter" else f"{run}-{adapter}"


def merge(run: str, adapter: str = "adapter", base_model: str = "Qwen/Qwen3.5-4B", log=print) -> Path:
    """The base model with the LoRA merged into its weights, saved like a normal model (bf16, on the CPU)."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    src = CHECKPOINTS / run / adapter
    out = CHECKPOINTS / run / ("merged" if adapter == "adapter" else f"merged-{adapter}")
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, src).merge_and_unload()
    model.save_pretrained(out)
    tok = AutoTokenizer.from_pretrained(base_model)
    tok.save_pretrained(out)
    if getattr(tok, "chat_template", None) is None:     # Qwen3.5 keeps it in the processor; GGUF wants it too
        from transformers import AutoProcessor

        (out / "chat_template.jinja").write_text(AutoProcessor.from_pretrained(base_model).chat_template,
                                                 encoding="utf-8")
    (out / "merged_from.json").write_text(json.dumps({"base_model": base_model, "adapter": str(src.relative_to(ROOT)),
                                                      "dtype": "bfloat16"}, indent=2), encoding="utf-8")
    log(f"merged: {out}")
    return out


def to_gguf(merged: Path, name: str, outtype: str = "bf16") -> Path:
    out = merged.parent / "gguf" / f"{name}-{outtype}.gguf"
    out.parent.mkdir(parents=True, exist_ok=True)
    # --no-mtp: the text-only model has no multi-token-prediction head (Qwen3.5's own MTP layers aren't loaded)
    subprocess.run([sys.executable, str(CONVERTER), str(merged), "--outfile", str(out), "--outtype", outtype,
                    "--no-mtp"], check=True)
    return out


def quantize(src: Path, kind: str) -> Path:
    """llama-quantize: bf16 GGUF -> Q8_0, Q4_K_M ... (kind as llama.cpp names it)."""
    out = src.with_name(src.name.replace("-bf16.gguf", f"-{kind.lower()}.gguf"))
    subprocess.run([str(QUANTIZE), str(src), str(out), kind], check=True)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Merge a clean-up adapter into Qwen3.5-4B and convert it for llama.cpp.")
    p.add_argument("--run", default="full1", help="checkpoints/cleanup/<run>/")
    p.add_argument("--adapter", default="adapter", help="adapter folder in the run: adapter (last epoch), epoch1 ...")
    p.add_argument("--quant", nargs="*", default=["Q8_0", "Q4_K_M"], help="llama-quantize types; none for bf16 only")
    args = p.parse_args(argv)
    for path in (CONVERTER, QUANTIZE):
        if not path.exists():
            raise SystemExit(f"Missing {path}: llama.cpp b11960 goes in {LLAMA_CPP} (see the docstring)")
    gguf = to_gguf(merge(args.run, args.adapter), label(args.run, args.adapter))
    print(f"GGUF: {gguf}")
    for kind in args.quant:
        print(f"GGUF: {quantize(gguf, kind)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
