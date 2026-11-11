"""Download student models from Hugging Face with byte-level progress."""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from fnmatch import fnmatch
from pathlib import Path

# Must be set before huggingface_hub is imported: its own progress bars
# would scribble over the TUI.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from huggingface_hub import HfApi, constants, snapshot_download  # noqa: E402
from huggingface_hub.errors import (  # noqa: E402
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
)

from . import state  # noqa: E402
from .catalog import Model  # noqa: E402

PATTERNS = ["*.json", "*.safetensors", "*.model", "*.tiktoken", "*.txt", "*.jinja"]

Progress = Callable[[int, int], None]  # (bytes_done, bytes_total)


class DownloadError(Exception):
    pass


def _wanted(filename: str) -> bool:
    return any(fnmatch(filename, p) for p in PATTERNS)


def _repo_cache(repo: str) -> Path:
    return Path(constants.HF_HUB_CACHE) / f"models--{repo.replace('/', '--')}"


def _dir_size(path: Path) -> int:
    total = 0
    if path.is_dir():
        for f in path.rglob("*"):
            try:
                if f.is_file() and not f.is_symlink():
                    total += f.stat().st_size
            except OSError:
                pass
    return total


def _gated_message(repo: str) -> str:
    return (
        f"{repo} is gated. Accept its license at https://huggingface.co/{repo}, "
        "then log in with `hf auth login` (or set HF_TOKEN)."
    )


def download(model: Model, on_progress: Progress | None = None) -> Path:
    try:
        info = HfApi().model_info(model.repo, files_metadata=True)
    except GatedRepoError as e:
        raise DownloadError(_gated_message(model.repo)) from e
    except RepositoryNotFoundError as e:
        raise DownloadError(f"Repo not found: {model.repo}") from e
    except HfHubHTTPError as e:
        raise DownloadError(f"Hugging Face error: {e}") from e
    except Exception as e:  # network down, DNS, etc.
        raise DownloadError(f"Could not reach Hugging Face: {e}") from e

    total = sum((s.size or 0) for s in (info.siblings or []) if _wanted(s.rfilename))
    blobs = _repo_cache(model.repo) / "blobs"
    stop = threading.Event()

    def poll() -> None:
        while not stop.wait(0.5):
            if on_progress:
                on_progress(min(_dir_size(blobs), total), total)

    watcher = threading.Thread(target=poll, daemon=True)
    watcher.start()
    try:
        path = snapshot_download(model.repo, allow_patterns=PATTERNS)
    except GatedRepoError as e:
        raise DownloadError(_gated_message(model.repo)) from e
    except HfHubHTTPError as e:
        raise DownloadError(f"Hugging Face error: {e}") from e
    except Exception as e:
        raise DownloadError(f"Download failed: {e}") from e
    finally:
        stop.set()
        watcher.join(timeout=2)

    if on_progress:
        on_progress(total, total)
    state.mark_downloaded(model.id, path)
    return Path(path)
