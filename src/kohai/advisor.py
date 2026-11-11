"""Turns detected hardware into guidance: what fits, what to pick, how to train it.

VRAM numbers are rough planning estimates for Unsloth-style QLoRA with gradient
checkpointing - good enough to steer choices, not a guarantee.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

from .catalog import Model
from .sysinfo import SystemInfo

# Fit verdicts
OK = "ok"  # fits with headroom
TIGHT = "tight"  # fits, but expect to tune batch/seq length
CLOUD = "cloud"  # needs a bigger (cloud) GPU
SLOW = "slow"  # possible on CPU, but slow
NO = "no"

# Run verdicts
RUN_GPU = "gpu"
RUN_CPU = "cpu"

DISPLAY_OVERHEAD_GB = 0.6  # VRAM the desktop/driver typically keeps
COMFORT = 0.85  # "ok" means using at most 85% of the budget
COLAB_T4_GB = 15.0  # free Colab GPU, used as the cloud reference


@dataclass(frozen=True)
class Estimate:
    infer_gb: float  # 4-bit inference
    train_gb: float  # QLoRA SFT
    rl_gb: float  # GRPO (training + generation)


def estimate(m: Model, seq_len: int = 1024) -> Estimate:
    p = m.params_b
    s = seq_len / 1024
    infer = 0.6 * p + 0.5
    train = 0.55 * p + 0.30 * p * s + 1.5
    rl = train + 0.5 * p + 0.8
    return Estimate(round(infer, 1), round(train, 1), round(rl, 1))


def _fit(need: float, budget: float) -> str:
    if budget > 0 and need <= budget * COMFORT:
        return OK
    if budget > 0 and need <= budget:
        return TIGHT
    return CLOUD


@dataclass
class Verdict:
    model: Model
    est: Estimate
    run: str
    train: str
    rl: str


@dataclass
class Advice:
    backend: str  # cuda | mps | cpu
    budget_gb: float  # usable accelerator memory
    verdicts: list[Verdict]
    recommended: Verdict | None = None
    stretch: Verdict | None = None
    fast: Verdict | None = None
    train_where: str = "local"  # local | colab
    settings_budget: float = 0.0  # memory the settings are computed for
    settings: dict = field(default_factory=dict)
    tips: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def badge(self, model_id: str) -> str:
        if self.recommended and self.recommended.model.id == model_id:
            return "★"
        if self.stretch and self.stretch.model.id == model_id:
            return "◆"
        if self.fast and self.fast.model.id == model_id:
            return "✦"
        return ""

    def settings_for(self, v: Verdict) -> dict:
        """Suggested training settings for any model (empty if it can't train here)."""
        if v.train == CLOUD and self.train_where == "local":
            return {}
        return _settings(v, self.settings_budget)

    def verdict(self, model_id: str) -> Verdict | None:
        return next((v for v in self.verdicts if v.model.id == model_id), None)


def compute_budget(info: SystemInfo) -> tuple[str, float]:
    gpu = info.best_gpu
    if gpu and gpu.vendor == "nvidia" and gpu.vram_gb:
        return "cuda", round(max(gpu.vram_gb - DISPLAY_OVERHEAD_GB, 0.0), 1)
    if gpu and gpu.vendor == "apple":
        return "mps", round(info.ram_gb * 0.6, 1)
    return "cpu", 0.0


def judge(m: Model, backend: str, budget: float, ram_gb: float) -> Verdict:
    est = estimate(m)
    cpu_ok = est.infer_gb <= ram_gb * 0.5
    if backend == "cpu":
        run = RUN_CPU if cpu_ok else NO
        train = SLOW if m.params_b <= 0.6 else CLOUD
        rl = CLOUD
    else:
        run = RUN_GPU if est.infer_gb <= budget else (RUN_CPU if cpu_ok else NO)
        train = _fit(est.train_gb, budget)
        rl = _fit(est.rl_gb, budget)
    return Verdict(m, est, run, train, rl)


def _rank(v: Verdict) -> tuple:
    # Better student first, then prefer no license gate, then bigger.
    return (v.model.quality, not v.model.gated, v.model.params_b)


def _pick(verdicts: list[Verdict]) -> tuple[Verdict | None, Verdict | None, Verdict | None]:
    rl_ok = [v for v in verdicts if v.rl == OK]
    train_ok = [v for v in verdicts if v.train == OK]
    recommended = max(rl_ok or train_ok, key=_rank, default=None)

    # Stretch: the best student that is bigger than the recommendation and still trains locally.
    floor = recommended.model.params_b if recommended else 0.0
    bigger = [v for v in verdicts if v.train in (OK, TIGHT) and v.model.params_b > floor]
    stretch = max(bigger, key=_rank, default=None)

    quick = [
        v
        for v in verdicts
        if v.train in (OK, SLOW) and v.model.quality >= 3 and v is not recommended
    ]
    fast = min(quick, key=lambda v: (v.model.params_b, -v.model.quality), default=None)
    return recommended, stretch, fast


def _settings(rec: Verdict | None, budget: float) -> dict:
    if rec is None:
        return {}
    seq = 512
    for candidate in (2048, 1024):
        if estimate(rec.model, candidate).train_gb <= budget * COMFORT:
            seq = candidate
            break
    batch = 1 if budget < 8 else 2 if budget < 16 else 4
    return {
        "method": "QLoRA (4-bit)" if budget < 20 else "LoRA (16-bit)",
        "max_seq_len": seq,
        "batch_size": batch,
        "grad_accum": max(16 // batch, 1),
        "lora_rank": 16 if budget < 12 else 32,
        "gradient_checkpointing": "unsloth",
        "rl_group_size": 4 if budget < 8 else 8,
    }


def advise(info: SystemInfo, catalog: list[Model]) -> Advice:
    backend, budget = compute_budget(info)
    verdicts = [judge(m, backend, budget, info.ram_gb) for m in catalog]
    advice = Advice(backend=backend, budget_gb=budget, verdicts=verdicts)

    if backend == "cpu":
        # Recommend for a free Colab T4 instead; local machine does the rest.
        advice.train_where = "colab"
        cloud = [judge(m, "cuda", COLAB_T4_GB, info.ram_gb) for m in catalog]
        rec, stretch, _ = _pick(cloud)
        advice.recommended = advice.verdict(rec.model.id) if rec else None
        advice.stretch = advice.verdict(stretch.model.id) if stretch else None
        advice.fast = next((v for v in verdicts if v.train == SLOW and v.model.quality >= 3), None)
        advice.settings_budget = COLAB_T4_GB
        advice.settings = _settings(rec, COLAB_T4_GB)
    else:
        advice.recommended, advice.stretch, advice.fast = _pick(verdicts)
        advice.settings_budget = budget
        advice.settings = _settings(advice.recommended, budget)

    _add_tips(info, advice)
    return advice


def _add_tips(info: SystemInfo, a: Advice) -> None:
    gpu = info.best_gpu
    tips, warn = a.tips, a.warnings

    if a.backend == "cuda" and gpu:
        tips.append(
            f"{a.budget_gb:.1f} GB usable VRAM (~{DISPLAY_OVERHEAD_GB} GB kept for your display)."
        )
        if a.budget_gb < 8:
            tips.append("Sweet spot for you: 0.5B-1.7B students. RL is where they beat big models.")
        elif a.budget_gb < 16:
            tips.append("You can RL-train up to ~3-4B models locally.")
        else:
            tips.append("Plenty of VRAM: 4B students with RL, or larger with QLoRA.")
        if gpu.is_laptop:
            tips.append("Laptop GPU: keep it plugged in. On battery it throttles hard.")
        if info.os == "Windows" and not info.in_wsl:
            tips.append(
                "Native Windows training works with Unsloth. If you hit triton/bitsandbytes "
                "errors, WSL2 is the fallback (`wsl --install -d Ubuntu`)."
            )
    elif a.backend == "mps":
        tips.append("Apple Silicon: budget is ~60% of unified memory. Training will use MLX.")
    else:
        if gpu and gpu.vendor in ("amd", "intel"):
            warn.append(
                f"{gpu.name}: the training stack needs an NVIDIA (CUDA) GPU. "
                "Run models on CPU here and train on Colab."
            )
        tips.append(
            "No CUDA GPU: train on a free Google Colab T4 (16 GB). kohai still runs teachers, "
            "data generation and evaluation locally."
        )

    if sys.version_info >= (3, 13) or info.python.startswith(("3.13", "3.14")):
        tips.append("Training env uses Python 3.12 for ML-library compatibility (uv handles it).")
    if info.ram_gb < 12:
        warn.append(f"Only {info.ram_gb:.0f} GB RAM: close other apps while training.")
    if info.disk_free_gb < 20:
        warn.append(
            f"Only {info.disk_free_gb:.0f} GB free disk: models, datasets and checkpoints add up fast."
        )
