"""Step 4c — pin the pipeline output to an INTERNAL snapshot, and generate the label summary.

Not a public release. The labels are auto-generated and only partially human-verified; the
snapshot exists so that reported numbers point at one exact, checkable state of the data.

The deliverable is `README.md` inside the bundle — the **label & tracking summary**: what each
stage did to the data, how much it kept, and what is known about the error it introduced. Every
figure in it is read from the artifacts and the run reports, so it cannot go stale. `--summary-to`
also drops a copy in `docs/` for the write-up.

Usage:
    python scripts/05_package_snapshot.py                    # build data/snapshots/zod-ped-v0.2
    python scripts/05_package_snapshot.py --summary-to docs/LABEL_SUMMARY.md
    python scripts/05_package_snapshot.py --overwrite --tarball
    python scripts/05_package_snapshot.py --verify data/snapshots/zod-ped-v0.2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _common import DEFAULT_SPLITS_PATH, DEFAULT_TRAJ_DIR, ROOT
from zodped.dataset.packaging import (SUMMARY_FILE, BundleSources, build_bundle, make_tarball,
                                      verify_bundle)
from zodped.dataset.stats import DEFAULT_HORIZON, HORIZONS

DEFAULT_VERSION = "0.2"
DEFAULT_OUT_ROOT = ROOT / "data" / "snapshots"
DOC_NAMES = ("DATA_FORMAT.md", "PIPELINE.md", "EXPERIMENTS_LOG.md", "JAAD_PIE_ALIGNMENT.md")


def default_sources(version: str) -> BundleSources:
    """Snapshot inputs at their canonical pipeline locations."""
    processed = ROOT / "data" / "processed"
    return BundleSources(
        annotations=ROOT / "data" / "annotations",
        splits=DEFAULT_SPLITS_PATH,
        schema=ROOT / "configs" / f"dataset_schema_v{version}.yaml",
        trajectories=DEFAULT_TRAJ_DIR,
        actions=processed / "actions",
        actions_verified=processed / "actions_verified",
        reports=processed / "reports",
        docs=[ROOT / "docs" / name for name in DOC_NAMES],
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", default=DEFAULT_VERSION,
                    help="snapshot version; also selects configs/dataset_schema_v<version>.yaml")
    ap.add_argument("--out", type=Path, default=None,
                    help="bundle directory (default: data/snapshots/zod-ped-v<version>)")
    ap.add_argument("--horizon", default=DEFAULT_HORIZON, choices=HORIZONS,
                    help="horizon the summary's headline tables use "
                         "(the manifest always carries all of them)")
    ap.add_argument("--summary-to", type=Path, default=None,
                    help="also write the label summary here, e.g. docs/LABEL_SUMMARY.md")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing bundle directory")
    ap.add_argument("--tarball", action="store_true", help="also write <bundle>.tar.gz")
    ap.add_argument("--verify", type=Path, default=None,
                    help="verify an existing bundle against its checksums and exit")
    args = ap.parse_args()

    if args.verify:
        problems = verify_bundle(args.verify)
        if problems:
            print(f"FAILED — {len(problems)} problem(s) in {args.verify}")
            for problem in problems:
                print(f"  {problem}")
            return 1
        print(f"OK — {args.verify} matches its checksums")
        return 0

    out_dir = args.out or DEFAULT_OUT_ROOT / f"zod-ped-v{args.version}"
    try:
        result = build_bundle(
            sources=default_sources(args.version),
            out_dir=out_dir,
            version=args.version,
            repo_root=ROOT,
            horizon=args.horizon,
            overwrite=args.overwrite,
        )
    except (FileExistsError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        return 1

    manifest = result.manifest
    summary = manifest["composition"][args.horizon]
    print(f"SNAPSHOT  {result.path}   [INTERNAL — not a release]")
    print(f"  version {manifest['version']}  commit {(manifest['git']['commit'] or '?')[:12]}"
          f"{'  [DIRTY TREE]' if manifest['git']['dirty'] else ''}")
    print(f"  {result.n_files} files, checksummed")
    print(f"  {summary['totals']['samples']} samples / {summary['totals']['pedestrians']} "
          f"pedestrians / {summary['totals']['sequences']} sequences  (horizon {args.horizon}s)")
    for tier, block in summary["by_tier"].items():
        print(f"    {tier:6} {block['samples']:>5} samples, {block['crossers']:>4} crossers "
              f"(ratio {block['crossing_ratio']:.3f})")

    if args.summary_to:
        args.summary_to.parent.mkdir(parents=True, exist_ok=True)
        args.summary_to.write_text((result.path / SUMMARY_FILE).read_text())
        print(f"  summary -> {args.summary_to}")
    if args.tarball:
        tar_path = make_tarball(result.path)
        print(f"  tarball {tar_path} ({tar_path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
