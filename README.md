# kohai

**Your tiny model's senpai is a big one.**

kohai turns a small local LLM (0.5B–4B) into a specialist that can beat models 10× its size
on one topic. It scans your hardware, tells you which small models you can actually train, downloads
the one you pick, and (soon) lets *any* AI be the teacher: Claude Code, OpenCode, Gemini CLI, or a
local model.

```bash
uvx --from git+https://github.com/imstillsamarth/kohai kohai
```

In Japanese dojo culture, the *kohai* (junior) learns from the *senpai* (senior). Here the senpai is
a big model, the kohai is your tiny one, and **Tofu** 🥋, a pixel tofu-cube sensei with a red
headband who lives in the TUI, runs the dojo. Tofu scans your machine, reacts to the models you pick,
cheers when downloads finish, and drops tips while you work.

## Why

A 27B model knows a bit of everything. A 1.5B model trained hard on **one** narrow, checkable
topic can beat it on that topic, and it runs on a laptop. The recipe:

1. **Distill:** the senpai writes thousands of examples on your topic
2. **Verify:** a checker throws away wrong ones (run the code, check the math)
3. **Fine-tune:** QLoRA on your GPU, even a 6 GB one
4. **RL (GRPO):** reward the kohai for verifiably correct answers
5. **Arena:** benchmark it head-to-head against a big model

## What works today (v0.1)

- **Hardware auto-detection**: OS, CPU, RAM, NVIDIA GPU + VRAM + CUDA, Apple Silicon, AMD/Intel, disk
- **Hardware-aware advice**: every model rated for *run / fine-tune / RL* on your machine
  (`● fits`, `◐ tight`, `○ cloud`), plus picks:
  - ★ **Recommended**: best student that fits fine-tuning *and* RL comfortably
  - ◆ **Stretch**: biggest good model you can still train
  - ✦ **Fast iteration**: tiny model to debug your dataset first
- **Suggested training settings** (seq length, batch, LoRA rank, RL group size) for your VRAM
- **One-key downloads** from Hugging Face with progress (gated models explained)
- No NVIDIA GPU? kohai plans for a free Colab T4 instead

## Usage

```bash
uv run kohai                   # TUI
uv run kohai scan              # hardware report + recommendations
uv run kohai scan --json       # machine-readable
uv run kohai download qwen2.5-1.5b
```

TUI keys: `↑/↓` select · `d` download · `f` RL-ready only · `r` rescan · `q` quit

Example on an RTX 3050 6 GB laptop:

```
NVIDIA CUDA · 5.4 GB training budget
★ Recommended: Qwen2.5 1.5B Instruct      (SFT + RL fit comfortably)
◆ Stretch:     Qwen3 4B                   (QLoRA fits, RL needs cloud)
✦ Fast:        Qwen2.5 0.5B Instruct
QLoRA (4-bit) · seq 2048 · batch 1×16 accum · LoRA r=16 · RL group 4
```

## Adding a model

Edit [`src/kohai/models.toml`](src/kohai/models.toml): copy a block, fill it in, open a PR.
VRAM estimates are computed from `params_b`.

## Roadmap

- [x] Hardware scan + model advisor + downloads
- [x] Tofu, the animated pixel sensei
- [ ] **Teacher adapters**: Claude Code (`claude -p`), OpenCode (`opencode run`), Gemini CLI,
      any OpenAI-compatible API (Ollama, OpenRouter, LM Studio)
- [ ] **Topic packs** with automatic checkers: PowerShell, regex, SQL, Countdown math, ciphers
- [ ] Dataset generation + verification
- [ ] Training: Unsloth QLoRA SFT → GRPO, using the advisor's settings
- [ ] Arena: kohai vs a big model on held-out questions
- [ ] Export: GGUF → Ollama in one command
- [ ] Browse and search every small model on Hugging Face, live

## A note on teachers

Some AI providers' terms restrict using their outputs to train competing models. kohai never ships
generated data: you pick the teacher and you're responsible for following its terms. Open-weight
teachers (Qwen, DeepSeek, Llama, Gemma via Ollama/OpenRouter) are the fully clean path.

## Development

```bash
uv sync
uv run pytest
```

VRAM numbers are planning estimates for Unsloth-style QLoRA with gradient checkpointing,
not guarantees. PRs with real measurements from your GPU are very welcome.

## License

[Apache-2.0](LICENSE)
