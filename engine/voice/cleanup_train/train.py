"""Trains a LoRA adapter for the dictation clean-up on Qwen3.5-4B (text only, thinking off).

    .\\.venv\\Scripts\\python.exe -m engine.voice.cleanup_train.train --max-steps 5      # smoke test, about a minute
    .\\.venv\\Scripts\\python.exe -m engine.voice.cleanup_train.train --name pilot       # data/cleanup_train/pilot.jsonl
    .\\.venv\\Scripts\\python.exe -m engine.voice.cleanup_train.train --name big2 --data big2_train --rank 32

Close the engine first: the 16 GB card fits one heavy job at a time.

Each example is the exact text the eval runner sends (the chat template with the generation prompt, thinking
off) followed by the answer and the end-of-turn token. Only the answer is learned: the prompt's tokens are
masked out of the loss. Plain PyTorch loop with peft (already in requirements-train.txt), micro-batch 1 (no
padding), gradient accumulation, gradient checkpointing, bf16.

Writes checkpoints/cleanup/<name>/: adapter/ (the LoRA weights after the last step, about 120 MB at r=16), epoch1/,
epoch2/ ... (the same after each earlier epoch), config.json, log.jsonl and results.json (loss before, after and
per epoch on held-back examples, time, peak GPU memory). checkpoints/ is git-ignored.

Over- or underfitting (big3 on): --held-back-by sentence keeps whole answers out of training, so the held-back
loss is on sentences never trained on (with "example", noisy copies of the same sentence land on both sides).
--watch <file> adds a loss on a file that is never trained on (bigset --nuance writes big1_test_watch.jsonl, 400
test cases). Every --check-every steps, and at each epoch's end, log.jsonl gets a row with "check": the loss on
300 training examples, on the held-back ones and on the watch file, all with dropout off. Training loss falling
while the other two rise is overfitting; all three high and flat is underfitting.

    .\\.venv\\Scripts\\python.exe -m engine.voice.cleanup_train.train --name big3 --data big3_train --rank 32 --epochs 2 --held-back-by sentence --watch big1_test_watch --check-every 500
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
    held_back_by: str = "example"  # "sentence": every example of some answers (see the docstring)
    watch: str = ""               # data/cleanup_train/<watch>.jsonl: a loss check on it too, never trained on
    check_every: int = 0          # steps between loss checks; 0 = only at each epoch's end
    max_steps: int = 0            # 0 = run all epochs
    save_epochs: bool = True      # also save the adapter after each epoch (epoch1/, epoch2/ ...)
    seed: int = 1234


CHECK_SIZE = 400                  # held-back and watch examples per loss check (a check takes about 2 minutes)
PROBE_SIZE = 300                  # training examples per loss check


def load_chats(name: str) -> list[dict]:
    with (DATA / f"{name}.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def answer_text(chat: dict) -> str:
    content = chat["messages"][-1]["content"]
    try:
        return json.loads(content)["text"].strip().lower()
    except (ValueError, KeyError, TypeError):
        return content.strip().lower()


def split_held_back(chats: list[dict], share: float, by: str, rng: random.Random) -> tuple[list[dict], list[dict]]:
    """(training, held back). `chats` is already shuffled. "example": the first share of them (as big1 and big2).
    "sentence": whole answers, in random order, until the share is reached, so no held-back answer is trained on."""
    n_val = max(1, round(len(chats) * share))
    if by == "example":
        return chats[n_val:], chats[:n_val]
    if by != "sentence":
        raise ValueError(f"held_back_by must be 'example' or 'sentence', not {by!r}")
    groups: dict[str, list[dict]] = {}
    for c in chats:
        groups.setdefault(answer_text(c), []).append(c)
    keys = list(groups)
    rng.shuffle(keys)
    held: set[str] = set()
    count = 0
    for k in keys:
        if count >= n_val:
            break
        held.add(k)
        count += len(groups[k])
    return [c for c in chats if answer_text(c) not in held], [c for c in chats if answer_text(c) in held]


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
    train_chats, val_chats = split_held_back(chats, cfg.held_back, cfg.held_back_by, rng)
    val = [encode(c, tok, template) for c in val_chats]
    data = [encode(c, tok, template) for c in train_chats]
    watch = [encode(c, tok, template) for c in load_chats(cfg.watch)] if cfg.watch else []
    log(f"{len(data)} training examples, {len(val)} held back (by {cfg.held_back_by}), {len(watch)} watched; "
        f"mean length {sum(len(d['input_ids']) for d in data) / len(data):.0f} tokens")

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

    def mean_loss(examples) -> float | None:
        if not examples:
            return None
        model.eval()
        with torch.no_grad():
            v = sum(loss_of(ex).item() for ex in examples) / len(examples)
        model.train()
        return round(v, 4)

    def val_loss() -> float:
        return mean_loss(val)

    checking = bool(cfg.watch) or cfg.check_every > 0 or cfg.held_back_by == "sentence"
    probe = data[:PROBE_SIZE]

    def check(step: int, f) -> dict:
        """Loss on training examples, held-back sentences and the watch file, dropout off; a row in log.jsonl."""
        t = time.perf_counter()
        row = {"step": step, "check": True, "train": mean_loss(probe), "held_back": mean_loss(val[:CHECK_SIZE]),
               "watch": mean_loss(watch[:CHECK_SIZE])}
        row["check_s"] = round(time.perf_counter() - t, 1)
        f.write(json.dumps(row) + "\n")
        f.flush()
        log(f"check at step {step}: train {row['train']} held back {row['held_back']} watch {row['watch']}")
        return row

    steps_per_epoch = max(1, len(data) // cfg.batch_size)
    total = cfg.max_steps or int(steps_per_epoch * cfg.epochs)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, min(cfg.warmup_steps, total // 2), total)
    before = val_loss()
    log(f"held-back loss before training: {before}")

    t0, order, per_epoch, checks = time.perf_counter(), [], {}, []
    model.train()
    with (out / "log.jsonl").open("w", encoding="utf-8") as f:
        if checking:
            checks.append(check(0, f))
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
            if checking and (step == total or step % steps_per_epoch == 0
                             or (cfg.check_every and step % cfg.check_every == 0)):
                checks.append(check(step, f))

    after = val_loss()
    model.save_pretrained(out / "adapter")
    results = {"examples": len(data), "held_back": len(val), "held_back_by": cfg.held_back_by,
               "watched": len(watch), "steps": total, "loss_before": before,
               "loss_after": after, "loss_per_epoch": per_epoch, "checks": checks,
               "minutes": round((time.perf_counter() - t0) / 60, 1),
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
    p.add_argument("--rank", type=int, default=16, help="LoRA rank r (big1: 16)")
    p.add_argument("--alpha", type=int, default=0, help="LoRA alpha; 0 = twice the rank")
    p.add_argument("--held-back-by", choices=["example", "sentence"], default="example",
                   help="hold back random examples (big1, big2) or whole sentences (big3)")
    p.add_argument("--watch", default="", help="data/cleanup_train/<watch>.jsonl: loss check only, never trained on")
    p.add_argument("--check-every", type=int, default=0, help="steps between loss checks (0 = each epoch's end)")
    args = p.parse_args(argv)
    cfg = TrainConfig(name=args.name, data=args.data, epochs=args.epochs, lr=args.lr, max_steps=args.max_steps,
                      lora_r=args.rank, lora_alpha=args.alpha or 2 * args.rank, held_back_by=args.held_back_by,
                      watch=args.watch, check_every=args.check_every)
    print(json.dumps(train(cfg), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
