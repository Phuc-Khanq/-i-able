"""Trains a LoRA adapter for the dictation clean-up on Qwen3.5-4B (text only, thinking off).

    .\\.venv\\Scripts\\python.exe -m engine.voice.cleanup_train.train --max-steps 5      # smoke test, about a minute
    .\\.venv\\Scripts\\python.exe -m engine.voice.cleanup_train.train --name pilot       # data/cleanup_train/pilot.jsonl

Close the engine first: the 16 GB card fits one heavy job at a time.

Each example is the exact text the eval runner sends (the chat template with the generation prompt, thinking
off) followed by the answer and the end-of-turn token. Only the answer is learned: the prompt's tokens are
masked out of the loss. Plain PyTorch loop with peft (already in requirements-train.txt), micro-batch 1 (no
padding), gradient accumulation, gradient checkpointing, bf16.

Writes checkpoints/cleanup/<name>/: adapter/ (the LoRA weights after the last step, about 120 MB at r=16), epoch1/,
epoch2/ ... (the same after each earlier epoch), config.json, log.jsonl and results.json (loss before, after and
per epoch on held-back examples, time, peak GPU memory). checkpoints/ is git-ignored.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "cleanup_train"
CHECKPOINTS = ROOT / "checkpoints" / "cleanup"


@dataclass
class TrainConfig:
    name: str = "pilot"
    data: str = "pilot"
    base_model: str = "Qwen/Qwen3.5-4B"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # Every linear layer of the attention, the linear-attention blocks and the MLP; not the tiny in_proj_a/b.
    target_modules: list = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv",
                                                          "in_proj_z", "out_proj", "gate_proj", "up_proj",
                                                          "down_proj"])
    lr: float = 2e-4
    epochs: float = 3.0
    batch_size: int = 8           # examples per optimizer step (gradient accumulation, micro-batch 1)
    warmup_steps: int = 10
    held_back: float = 0.05       # share of the examples kept out of training, for a loss check
    max_steps: int = 0            # 0 = run all epochs
    save_epochs: bool = True      # also save the adapter after each epoch (epoch1/, epoch2/ ...)
    seed: int = 1234


def load_chats(name: str) -> list[dict]:
    with (DATA / f"{name}.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def encode(chat: dict, tok, template) -> dict:
    """Token ids and labels: -100 on the prompt, the answer's ids on the answer and its end-of-turn."""
    prompt_msgs, answer = chat["messages"][:-1], chat["messages"][-1]["content"]
    prompt = template.apply_chat_template(prompt_msgs, add_generation_prompt=True, tokenize=False,
                                          enable_thinking=False)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(answer + "<|im_end|>", add_special_tokens=False)["input_ids"]
    return {"input_ids": p_ids + a_ids, "labels": [-100] * len(p_ids) + a_ids}


def train(cfg: TrainConfig, log=print) -> dict:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    out = CHECKPOINTS / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)

    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    template = tok
    if getattr(tok, "chat_template", None) is None:
        from transformers import AutoProcessor

        template = AutoProcessor.from_pretrained(cfg.base_model)
    chats = load_chats(cfg.data)
    rng.shuffle(chats)
    n_val = max(1, round(len(chats) * cfg.held_back))
    val = [encode(c, tok, template) for c in chats[:n_val]]
    data = [encode(c, tok, template) for c in chats[n_val:]]
    log(f"{len(data)} training examples, {len(val)} held back; mean length "
        f"{sum(len(d['input_ids']) for d in data) / len(data):.0f} tokens")

    torch.cuda.reset_peak_memory_stats()
    model, info = AutoModelForCausalLM.from_pretrained(cfg.base_model, dtype=torch.bfloat16, output_loading_info=True)
    if info.get("missing_keys"):
        raise RuntimeError(f"{len(info['missing_keys'])} weights missing when loading {cfg.base_model}")
    model.to("cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                                             target_modules=cfg.target_modules, task_type="CAUSAL_LM"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"trainable parameters: {trainable / 1e6:.1f} M")

    def loss_of(ex) -> torch.Tensor:
        ids = torch.tensor([ex["input_ids"]], device="cuda")
        labels = torch.tensor([ex["labels"]], device="cuda")
        return model(input_ids=ids, labels=labels).loss

    def val_loss() -> float:
        model.eval()
        with torch.no_grad():
            v = sum(loss_of(ex).item() for ex in val) / len(val)
        model.train()
        return round(v, 4)

    steps_per_epoch = max(1, len(data) // cfg.batch_size)
    total = cfg.max_steps or int(steps_per_epoch * cfg.epochs)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, min(cfg.warmup_steps, total // 2), total)
    before = val_loss()
    log(f"held-back loss before training: {before}")

    t0, order, per_epoch = time.perf_counter(), [], {}
    model.train()
    with (out / "log.jsonl").open("w", encoding="utf-8") as f:
        for step in range(1, total + 1):
            running = 0.0
            for _ in range(cfg.batch_size):
                if not order:
                    order = list(range(len(data)))
                    rng.shuffle(order)
                loss = loss_of(data[order.pop()]) / cfg.batch_size
                loss.backward()
                running += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            row = {"step": step, "loss": round(running, 4), "lr": sched.get_last_lr()[0],
                   "s": round(time.perf_counter() - t0, 1)}
            f.write(json.dumps(row) + "\n")
            f.flush()
            if step == 1 or step % 10 == 0 or step == total:
                log(f"step {step}/{total} loss {running:.3f} ({row['s']:.0f} s)")
            if cfg.save_epochs and step % steps_per_epoch == 0 and step < total:
                epoch = step // steps_per_epoch            # a copy per epoch, so dev can pick the best one
                model.save_pretrained(out / f"epoch{epoch}")
                per_epoch[f"epoch{epoch}"] = val_loss()
                log(f"saved epoch{epoch}, held-back loss {per_epoch[f'epoch{epoch}']}")

    after = val_loss()
    model.save_pretrained(out / "adapter")
    results = {"examples": len(data), "held_back": len(val), "steps": total, "loss_before": before,
               "loss_after": after, "loss_per_epoch": per_epoch, "minutes": round((time.perf_counter() - t0) / 60, 1),
               "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2), "adapter": str(out / "adapter")}
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    log(f"held-back loss after training: {after}")
    return results


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train the clean-up LoRA adapter on Qwen3.5-4B.")
    p.add_argument("--name", default="pilot", help="run name: checkpoints/cleanup/<name>/")
    p.add_argument("--data", default="pilot", help="data/cleanup_train/<data>.jsonl")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-steps", type=int, default=0, help="stop after N optimizer steps (smoke test)")
    args = p.parse_args(argv)
    cfg = TrainConfig(name=args.name, data=args.data, epochs=args.epochs, lr=args.lr, max_steps=args.max_steps)
    print(json.dumps(train(cfg), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
