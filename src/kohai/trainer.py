"""Training: SFT -> GRPO -> eval, as small hand-written loops on top of PyTorch + PEFT.

Runs in a subprocess (`python -m kohai.trainer <stage> <run-id>`) and reports progress as
JSON lines on stdout, so the CLI and the TUI can both drive it.

Memory notes (tuned for 6 GB cards):
- LoRA on every linear layer, base in bf16 or 4-bit NF4 (QLoRA) depending on the advisor.
- Gradient checkpointing always on.
- RL computes logits only for completion tokens (`logits_to_keep`), never the whole sequence.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import sys
import time

from .runs import Run, load_run
from .topics import Topic, get_topic

END_OF_TURN = ("<|im_end|>", "<end_of_turn>", "<|eot_id|>", "<|end|>", "<|endoftext|>")


def emit(event: dict) -> None:
    print(json.dumps(event), flush=True)


# ---------------------------------------------------------------------------
# model loading
# ---------------------------------------------------------------------------


def _dtype():
    import torch

    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def load(run: Run, adapters: tuple[str, ...] = (), trainable: bool = False):
    """Load the student (+ optional LoRA adapters). Returns (model, tokenizer, device)."""
    import torch
    import transformers
    from packaging.version import Version
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = _dtype()
    tok = AutoTokenizer.from_pretrained(run.model_repo)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs: dict = {}
    dtype_kw = "dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"
    kwargs[dtype_kw] = dtype
    quant = device == "cuda" and run.settings.get("method", "").startswith("QLoRA")
    if quant:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    if device == "cuda":
        kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(run.model_repo, **kwargs)

    if trainable:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()

    from peft import LoraConfig, PeftModel, get_peft_model

    if adapters:
        first, *rest = adapters
        # A trainable adapter must be named "default", or save_pretrained() nests it in a subfolder.
        model = PeftModel.from_pretrained(
            model, run.path(f"adapter-{first}"), adapter_name="default" if trainable else first,
            is_trainable=trainable,
        )
        for name in rest:
            model.load_adapter(run.path(f"adapter-{name}"), adapter_name=name)
    elif trainable:
        r = int(run.settings.get("lora_rank", 16))
        model = get_peft_model(
            model,
            LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM"),
        )
    return model, tok, device


def eos_ids(tok, model) -> list[int]:
    ids = set()
    if tok.eos_token_id is not None:
        ids.add(tok.eos_token_id)
    gen_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if isinstance(gen_eos, int):
        ids.add(gen_eos)
    elif isinstance(gen_eos, list):
        ids.update(gen_eos)
    for t in END_OF_TURN:
        tid = tok.convert_tokens_to_ids(t)
        if isinstance(tid, int) and tid != tok.unk_token_id and tid >= 0:
            ids.add(tid)
    return sorted(ids)


def chat_prefix(tok, user: str) -> str:
    msgs = [{"role": "user", "content": user}]
    # enable_thinking=False makes Qwen3 answer directly; other templates ignore it.
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def _turn_suffix(tok) -> str:
    probe = "KOHAI_PROBE_ANSWER"
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": probe}]
    try:
        full = tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
        suffix = full[full.rfind(probe) + len(probe) :]
        if suffix.strip():
            return suffix
    except Exception:
        pass
    return tok.eos_token or ""


def generate(model, tok, prompts: list[str], device: str, stops: list[int], max_new_tokens: int,
             sample: bool = False, temperature: float = 1.0):
    """Batched generation. Returns (sequences, prompt_len, completion_mask, texts)."""
    import torch

    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    gen_kwargs = dict(max_new_tokens=max_new_tokens, eos_token_id=stops, pad_token_id=tok.pad_token_id)
    if sample:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=1.0, top_k=0)
    else:
        gen_kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
    with torch.no_grad():
        seq = model.generate(**enc, **gen_kwargs)
    plen = enc.input_ids.shape[1]
    comp = seq[:, plen:]
    is_stop = torch.isin(comp, torch.tensor(stops, device=comp.device))
    has_stop = is_stop.any(dim=1)
    first = torch.where(has_stop, is_stop.int().argmax(dim=1), torch.full_like(has_stop, comp.shape[1] - 1, dtype=torch.long))
    cmask = torch.arange(comp.shape[1], device=comp.device)[None, :] <= first[:, None]
    texts = [tok.decode(comp[i][: int(first[i]) + 1], skip_special_tokens=True) for i in range(comp.shape[0])]
    return seq, enc.attention_mask, cmask, texts


# ---------------------------------------------------------------------------
# SFT
# ---------------------------------------------------------------------------


def sft(run: Run, topic: Topic, epochs: int | None = None, lr: float = 2e-4, max_len: int = 512) -> None:
    import torch

    examples = run.read_jsonl("train.jsonl")
    if not examples:
        raise SystemExit("No training data. Run `kohai gen` first.")
    model, tok, device = load(run, trainable=True)
    suffix = _turn_suffix(tok)

    rows = []
    for ex in examples:
        prefix = chat_prefix(tok, topic.prompt(ex))
        p_ids = tok(prefix, add_special_tokens=False).input_ids
        f_ids = tok(prefix + topic.target(ex) + suffix, add_special_tokens=False).input_ids
        if len(f_ids) <= max_len and len(f_ids) > len(p_ids):
            rows.append((f_ids, [-100] * len(p_ids) + f_ids[len(p_ids) :]))

    micro = int(run.settings.get("batch_size", 1))
    if run.params_b <= 1.0:
        micro = max(micro, 4)
    accum = max(1, 16 // micro)
    epochs = epochs or (3 if len(rows) < 400 else 2)
    per_epoch = math.ceil(len(rows) / (micro * accum))
    total = per_epoch * epochs
    warm = max(1, total // 20)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.05, (total - s) / max(1, total - warm))
    )
    emit({"event": "start", "stage": "sft", "examples": len(rows), "total": total, "epochs": epochs})

    model.train()
    step, t0, running = 0, time.time(), None
    autocast = torch.autocast("cuda", dtype=_dtype()) if device == "cuda" else contextlib.nullcontext()
    for epoch in range(epochs):
        random.shuffle(rows)
        micro_i = 0
        for i in range(0, len(rows), micro):
            chunk = rows[i : i + micro]
            width = max(len(r[0]) for r in chunk)
            ids = torch.full((len(chunk), width), tok.pad_token_id, dtype=torch.long)
            labels = torch.full((len(chunk), width), -100, dtype=torch.long)
            attn = torch.zeros((len(chunk), width), dtype=torch.long)
            for j, (f, lab) in enumerate(chunk):
                ids[j, : len(f)] = torch.tensor(f)
                labels[j, : len(lab)] = torch.tensor(lab)
                attn[j, : len(f)] = 1
            with autocast:
                out = model(input_ids=ids.to(device), attention_mask=attn.to(device), labels=labels.to(device))
            (out.loss / accum).backward()
            loss = float(out.loss)
            running = loss if running is None else 0.9 * running + 0.1 * loss
            micro_i += 1
            last = i + micro >= len(rows)
            if micro_i % accum == 0 or last:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                emit({
                    "event": "step", "stage": "sft", "step": step, "total": total, "epoch": epoch + 1,
                    "loss": round(running, 4), "lr": sched.get_last_lr()[0],
                    "elapsed": round(time.time() - t0, 1), "vram_gb": _vram(),
                })
    model.save_pretrained(run.path("adapter-sft"))
    emit({"event": "done", "stage": "sft", "elapsed": round(time.time() - t0, 1)})


# ---------------------------------------------------------------------------
# RL (GRPO-style: group-relative advantages, on-policy, token-level loss, no KL)
# ---------------------------------------------------------------------------


def rl(run: Run, topic: Topic, steps: int = 80, lr: float = 2e-5, max_new_tokens: int = 64) -> None:
    import torch

    examples = run.read_jsonl("train.jsonl")
    start_from = ("sft",) if run.path("adapter-sft").exists() else ()
    model, tok, device = load(run, adapters=start_from, trainable=True)
    stops = eos_ids(tok, model)
    group = int(run.settings.get("rl_group_size", 4))
    per_step = 2 if run.params_b > 1.0 else 4
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    autocast = torch.autocast("cuda", dtype=_dtype()) if device == "cuda" else contextlib.nullcontext()
    emit({"event": "start", "stage": "rl", "total": steps, "group": group, "prompts_per_step": per_step})

    t0 = time.time()
    avg_reward = avg_solved = None
    for step in range(1, steps + 1):
        batch = random.sample(examples, min(per_step, len(examples)))
        prompts = [chat_prefix(tok, topic.prompt(ex)) for ex in batch for _ in range(group)]
        exs = [ex for ex in batch for _ in range(group)]

        model.eval()
        seq, pmask, cmask, texts = generate(model, tok, prompts, device, stops, max_new_tokens, sample=True)
        rewards = torch.tensor([topic.reward(ex, t) for ex, t in zip(exs, texts)], dtype=torch.float32)
        solved = sum(topic.score(ex, t) >= 1.0 for ex, t in zip(exs, texts)) / len(texts)

        r = rewards.view(-1, group)
        std = r.std(dim=1, keepdim=True)
        adv = ((r - r.mean(dim=1, keepdim=True)) / (std + 1e-4)).masked_fill(std < 1e-6, 0.0).view(-1)

        loss_val = 0.0
        if adv.abs().sum() > 0:
            model.train()
            plen = pmask.shape[1]
            comp = seq[:, plen:]
            attn = torch.cat([pmask, cmask.long()], dim=1)
            pos = (attn.cumsum(-1) - 1).clamp(min=0)
            with autocast:
                try:
                    logits = model(input_ids=seq, attention_mask=attn, position_ids=pos,
                                   logits_to_keep=comp.shape[1] + 1).logits[:, :-1]
                except TypeError:
                    logits = model(input_ids=seq, attention_mask=attn, position_ids=pos).logits[:, plen - 1 : -1]
            logp = torch.log_softmax(logits.float(), dim=-1).gather(-1, comp.unsqueeze(-1)).squeeze(-1)
            m = cmask.float()
            loss = -(adv.to(device)[:, None] * logp * m).sum() / m.sum().clamp(min=1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            loss_val = float(loss)
            del logits, logp, loss

        mean_r = float(rewards.mean())
        avg_reward = mean_r if avg_reward is None else 0.9 * avg_reward + 0.1 * mean_r
        avg_solved = solved if avg_solved is None else 0.9 * avg_solved + 0.1 * solved
        emit({
            "event": "step", "stage": "rl", "step": step, "total": steps,
            "reward": round(avg_reward, 4), "solved": round(avg_solved, 4), "loss": round(loss_val, 4),
            "elapsed": round(time.time() - t0, 1), "vram_gb": _vram(),
            "sample": texts[0][:120],
        })
    model.save_pretrained(run.path("adapter-rl"))
    emit({"event": "done", "stage": "rl", "elapsed": round(time.time() - t0, 1)})


# ---------------------------------------------------------------------------
# Eval: base vs sft vs rl on the held-out test set
# ---------------------------------------------------------------------------


def evaluate(run: Run, topic: Topic, max_new_tokens: int = 96, batch_size: int = 8) -> dict:
    tests = run.read_jsonl("test.jsonl")
    if not tests:
        raise SystemExit("No test data.")
    trained = tuple(n for n in ("sft", "rl") if run.path(f"adapter-{n}").exists())
    model, tok, device = load(run, adapters=trained, trainable=False)
    model.eval()
    stops = eos_ids(tok, model)
    variants = ["base", *trained]
    emit({"event": "start", "stage": "eval", "total": len(tests) * len(variants), "variants": variants})

    results, samples, done = {}, {}, 0
    for variant in variants:
        if variant == "base" and trained:
            ctx = model.disable_adapter()
        else:
            if trained:
                model.set_adapter(variant)
            ctx = contextlib.nullcontext()
        scores, outputs = [], []
        with ctx:
            for i in range(0, len(tests), batch_size):
                chunk = tests[i : i + batch_size]
                prompts = [chat_prefix(tok, topic.prompt(ex)) for ex in chunk]
                _, _, _, texts = generate(model, tok, prompts, device, stops, max_new_tokens)
                for ex, t in zip(chunk, texts):
                    scores.append(topic.score(ex, t))
                    outputs.append(t.strip())
                done += len(chunk)
                emit({"event": "step", "stage": "eval", "step": done, "total": len(tests) * len(variants),
                      "variant": variant})
        results[variant] = {
            "solved": round(sum(s >= 1.0 for s in scores) / len(scores), 4),
            "score": round(sum(scores) / len(scores), 4),
            "n": len(scores),
        }
        samples[variant] = outputs[:5]
        emit({"event": "result", "stage": "eval", "variant": variant, **results[variant]})

    run.save_metrics({"eval": results, "samples": samples})
    emit({"event": "done", "stage": "eval", "results": results})
    return results


def _vram() -> float:
    try:
        import torch

        return round(torch.cuda.max_memory_allocated() / 1024**3, 2) if torch.cuda.is_available() else 0.0
    except Exception:
        return 0.0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m kohai.trainer")
    ap.add_argument("stage", choices=["sft", "rl", "eval"])
    ap.add_argument("run")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    random.seed(args.seed)
    try:
        import torch

        torch.manual_seed(args.seed)
    except ImportError:
        emit({"event": "error", "error": "Training needs the extras: uv sync --extra train"})
        return 2

    run = load_run(args.run)
    topic = get_topic(run.topic)
    try:
        if args.stage == "sft":
            sft(run, topic, epochs=args.epochs)
        elif args.stage == "rl":
            rl(run, topic, steps=args.steps)
        else:
            evaluate(run, topic)
    except Exception as e:  # report, then let the traceback land in the log
        emit({"event": "error", "stage": args.stage, "error": f"{type(e).__name__}: {e}"})
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
