"""Hardware auto-detection: OS, CPU, RAM, GPUs/VRAM, disk, Python."""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import psutil

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


@dataclass
class GPU:
    name: str
    vendor: str  # nvidia | amd | intel | apple | unknown
    vram_gb: float | None = None  # None = unknown or unified memory
    driver: str | None = None
    cuda: str | None = None
    compute_capability: str | None = None

    @property
    def is_laptop(self) -> bool:
        return "laptop" in self.name.lower() or "mobile" in self.name.lower()


@dataclass
class SystemInfo:
    os: str
    os_release: str
    arch: str
    cpu: str
    cores: int
    threads: int
    ram_gb: float
    disk_free_gb: float
    python: str
    in_wsl: bool
    gpus: list[GPU] = field(default_factory=list)

    @property
    def best_gpu(self) -> GPU | None:
        nvidia = [g for g in self.gpus if g.vendor == "nvidia"]
        if nvidia:
            return max(nvidia, key=lambda g: g.vram_gb or 0)
        apple = [g for g in self.gpus if g.vendor == "apple"]
        if apple:
            return apple[0]
        return self.gpus[0] if self.gpus else None

    def to_dict(self) -> dict:
        return asdict(self)


def _run(cmd: list[str], timeout: float = 6.0) -> str | None:
    exe = shutil.which(cmd[0])
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, *cmd[1:]],
            capture_output=True,
            text=True,
            timeout=timeout,
            creationflags=_NO_WINDOW,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def _vendor_from_name(name: str) -> str:
    n = name.lower()
    if "nvidia" in n or "geforce" in n or "rtx" in n or "quadro" in n:
        return "nvidia"
    if "amd" in n or "radeon" in n:
        return "amd"
    if "intel" in n:
        return "intel"
    if "apple" in n:
        return "apple"
    return "unknown"


def _detect_nvidia() -> list[GPU]:
    fields = "name,memory.total,driver_version,compute_cap"
    out = _run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"])
    if out is None:  # older drivers don't know compute_cap
        out = _run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"]
        )
    if not out:
        return []

    cuda = None
    header = _run(["nvidia-smi"])
    if header:
        m = re.search(r"CUDA Version:\s*([\d.]+)", header)
        if m:
            cuda = m.group(1)

    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            vram = round(float(parts[1]) / 1024, 1)
        except ValueError:
            vram = None
        gpus.append(
            GPU(
                name=parts[0],
                vendor="nvidia",
                vram_gb=vram,
                driver=parts[2],
                cuda=cuda,
                compute_capability=parts[3] if len(parts) > 3 else None,
            )
        )
    return gpus


def _detect_other_gpus() -> list[GPU]:
    system = platform.system()
    if system == "Darwin":
        if platform.machine() == "arm64":
            chip = _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or "Apple Silicon"
            return [GPU(name=f"{chip} GPU", vendor="apple")]
        return []
    if system == "Windows":
        out = _run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name",
            ],
            timeout=10,
        )
        names = [n.strip() for n in (out or "").splitlines() if n.strip()]
        return [GPU(name=n, vendor=_vendor_from_name(n)) for n in names]
    if system == "Linux":
        out = _run(["lspci", "-mm"])
        gpus = []
        for line in (out or "").splitlines():
            if "VGA" in line or "3D controller" in line:
                name = line.split('"')[5] if line.count('"') >= 6 else line
                gpus.append(GPU(name=name, vendor=_vendor_from_name(line)))
        return gpus
    return []


def _cpu_name() -> str:
    system = platform.system()
    if system == "Windows":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            ) as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    elif system == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    elif system == "Darwin":
        name = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if name:
            return name
    return platform.processor() or platform.machine()


def model_cache_dir() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return home / "hub"


def _disk_free_gb(path: Path) -> float:
    p = path
    while not p.exists() and p != p.parent:
        p = p.parent
    try:
        return round(shutil.disk_usage(p).free / 1024**3, 1)
    except OSError:
        return 0.0


def scan() -> SystemInfo:
    system = platform.system()
    release = platform.release()
    if system == "Windows":
        release = f"{release} (build {platform.version()})"

    gpus = _detect_nvidia()
    if not gpus:
        gpus = _detect_other_gpus()

    return SystemInfo(
        os=system,
        os_release=release,
        arch=platform.machine(),
        cpu=_cpu_name(),
        cores=psutil.cpu_count(logical=False) or 1,
        threads=psutil.cpu_count(logical=True) or 1,
        ram_gb=round(psutil.virtual_memory().total / 1024**3, 1),
        disk_free_gb=_disk_free_gb(model_cache_dir()),
        python=platform.python_version(),
        in_wsl="microsoft" in platform.release().lower(),
        gpus=gpus,
    )


if __name__ == "__main__":  # pragma: no cover
    import json

    json.dump(scan().to_dict(), sys.stdout, indent=2)
