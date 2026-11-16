"""Freeze a built benchmark into a versioned, hash-manifested release.

The narrative text of a built benchmark already lives in ``records.json``, but
that artifact sits under the git-ignored ``results/`` tree and is regenerated
(non-deterministically, at ``temperature`` 0.8) on every build. To make the
benchmark a *citable, reproducible* object, ``freeze`` copies one canonical build
into a committable directory (default ``<project_root>/benchmark_release/``) and
writes a ``MANIFEST.json`` pinning a SHA-256 of every file plus the git commit,
seed, and generator identities it was produced with.

Downstream experiments then load this frozen copy with **zero** API calls — for
you and for anyone reproducing the paper — and the manifest hashes let a reader
verify they have the exact bytes the results were computed on. For an external
release, deposit the frozen directory (e.g. Zenodo) and cite the DOI.

Run with::

    python -m msi_bench.experiments.run freeze
    python -m msi_bench.experiments.run freeze --from benchmark_degraded --to benchmark_release_degraded
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from ..config import Config, get_config

# Files emitted by ``Benchmark.save``; copied verbatim if present.
# ``attack_reserve.json`` is optional — present when the build
# generated the fixed-genuine pressure-sweep pool.
_BENCH_FILES = ("records.json", "events.json", "histories.json",
                "meta.json", "DATASHEET.md", "attack_reserve.json")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit(root: Path) -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(root),
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:  # noqa: BLE001 - git absence is non-fatal
        return None


def freeze_dir(src_dir: Path, dst_dir: Path, git_root: Optional[Path] = None) -> Path:
    """Copy a benchmark directory to ``dst_dir`` and write ``MANIFEST.json``.

    Pure filesystem logic (no :class:`Config` dependency) so it is unit-testable
    in isolation. Returns ``dst_dir``.
    """
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    if not src_dir.is_dir():
        raise FileNotFoundError(f"benchmark source not found: {src_dir}")
    dst_dir.mkdir(parents=True, exist_ok=True)

    files = {}
    for name in _BENCH_FILES:
        sp = src_dir / name
        if not sp.is_file():
            continue
        shutil.copy2(sp, dst_dir / name)
        files[name] = {"sha256": _sha256(dst_dir / name),
                       "bytes": (dst_dir / name).stat().st_size}
    if "records.json" not in files:
        raise FileNotFoundError(f"{src_dir} does not look like a built benchmark "
                                f"(no records.json)")

    meta = {}
    if (src_dir / "meta.json").is_file():
        meta = json.loads((src_dir / "meta.json").read_text("utf-8"))

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(git_root or src_dir),
        "source": str(src_dir),
        "seed": meta.get("seed"),
        "n_events": meta.get("n_events"),
        "total_records": meta.get("total_records"),
        "generators": meta.get("generators"),
        "degraded": meta.get("degraded"),
        "files": files,
    }
    (dst_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), "utf-8")
    return dst_dir


def run(cfg: Config, src: str = "benchmark", dst: Optional[str] = None) -> Path:
    src_dir = cfg.output_dir / src
    if dst:
        dst_dir = Path(dst)
        if not dst_dir.is_absolute():
            dst_dir = cfg.project_root / dst_dir
    else:
        dst_dir = cfg.project_root / "benchmark_release"
    out = freeze_dir(src_dir, dst_dir, git_root=cfg.project_root)
    manifest = json.loads((out / "MANIFEST.json").read_text("utf-8"))
    print(f"[freeze] froze {src_dir} -> {out}")
    print(f"[freeze] {manifest['total_records']} records, {manifest['n_events']} events, "
          f"seed={manifest['seed']}, commit={manifest['git_commit']}")
    print(f"[freeze] manifest: {out / 'MANIFEST.json'}")
    print("[freeze] commit this directory, or deposit it (e.g. Zenodo) and cite the DOI; "
          "experiments load it with --benchmark, and `run` falls back to it automatically.")
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Freeze a built benchmark.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--from", dest="src", default="benchmark",
                    help="Source subdir under output_dir (default: benchmark).")
    ap.add_argument("--to", dest="dst", default=None,
                    help="Destination dir (default: <project_root>/benchmark_release).")
    args = ap.parse_args(argv)
    cfg = get_config(args.config)
    try:
        run(cfg, src=args.src, dst=args.dst)
    except FileNotFoundError as exc:
        sys.exit(f"[freeze] {exc}. Build a benchmark first "
                 f"(python -m msi_bench.experiments.run build).")


if __name__ == "__main__":
    main()
