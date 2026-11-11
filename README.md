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

## What works today

**Pick the right student**
- **Hardware auto-detection**: OS, CPU, RAM, NVIDIA GPU + VRAM + CUDA, Apple Silicon, AMD/Intel, disk
- **Hardware-aware advice**: every model rated for *run / fine-tune / RL* on your machine
  (`● fits`, `◐ tight`, `○ cloud`), plus picks:
  - ★ **Recommended**: best student that fits fine-tuning *and* RL comfortably
  - ◆ **Stretch**: biggest good model you can still train
  - ✦ **Fast iteration**: tiny model to debug your dataset first
- **Suggested training settings** for your VRAM (16-bit LoRA when it fits, 4-bit QLoRA when it doesn't)
- **One-key downloads** from Hugging Face with progress

**Train it**
- **Any AI as senpai**: Claude Code, OpenCode, Gemini CLI, Codex CLI, or any OpenAI-compatible
  server (Ollama, LM Studio, OpenRouter)
- **Verified data**: the senpai writes tasks with reference answers; kohai runs every answer
  through the topic's checker and throws away the wrong ones
- **SFT → RL → exam**: LoRA fine-tuning, then GRPO-style RL against the checker, then a held-out
  exam that compares the model before and after
- **Arena**: put any big model through the same exam
- **Tofu trains with you** in the TUI: live progress, losses, rewards and a final scoreboard

## Usage

```bash
uv sync --extra train            # one-time: PyTorch (CUDA) + Transformers + PEFT
uv run kohai                     # TUI: pick a model, press t to train it
```

Or step by step from the CLI:

```bash
uv run kohai scan                                    # what can my machine train?
uv run kohai teachers --test                         # which senpai work here?
uv run kohai gen --topic regex --teacher claude -n 300
uv run kohai train                                   # SFT → RL → exam on the latest run
uv run kohai arena --vs opencode                     # same exam for a big model
uv run kohai runs                                    # all runs and their scores
```

TUI keys: `↑/↓` select · `t` train · `d` download · `f` RL-ready only · `r` rescan · `q` quit

Example on an RTX 3050 6 GB laptop:

```
NVIDIA CUDA · 5.4 GB training budget
★ Recommended: Qwen2.5 1.5B Instruct      (SFT + RL fit comfortably)
◆ Stretch:     Qwen3 4B                   (QLoRA fits, RL needs cloud)
✦ Fast:        Qwen2.5 0.5B Instruct
```

## Topic packs

| id | the kohai learns to | checker |
|----|---------------------|---------|
| `regex` | write a regex from a plain-English request + examples | runs it against hidden match / no-match strings |

A topic pack is one small Python class (`src/kohai/topics/`): a prompt asking the senpai for tasks,
a validator, the student prompt, and a scoring function. PRs for new topics are very welcome.

## Adding a model

Edit [`src/kohai/models.toml`](src/kohai/models.toml): copy a block, fill it in, open a PR.
VRAM estimates are computed from `params_b`.

## Roadmap

- [x] Hardware scan + model advisor + downloads
- [x] Tofu, the animated pixel sensei
- [x] Teacher adapters: Claude Code, OpenCode, Gemini CLI, Codex, OpenAI-compatible APIs
- [x] Dataset generation + verification
- [x] Training: LoRA/QLoRA SFT → GRPO, using the advisor's settings
- [x] Arena: kohai vs a big model on held-out questions
- [ ] More topic packs: SQL, Countdown math, shell commands (sandboxed), ciphers
- [ ] Chat with your trained kohai
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
