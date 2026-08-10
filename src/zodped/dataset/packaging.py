"""Build a checksummed INTERNAL snapshot of the pipeline output, and the label summary (Step 4c).

**This is not a public release.** The labels are auto-generated and only partially human-verified,
so a snapshot exists for one reason: to pin a set of reported numbers to one exact state of the
data. Without it, "GOLD is 17.3% crossers" describes nothing checkable — re-run the pipeline and
nobody can tell whether the pipeline moved or the number was always loose.

The snapshot's README is the **label & tracking summary** (`build_summary`): what each stage did
to the data, how many units it kept, and what is known about the error it introduced. That
document is the deliverable; the bundle around it is what makes it verifiable.

Two rules shape the layout:

  * **The annotations directory keeps its shape.** The bundle's `annotations/` is a copy of
    `data/annotations/`, so `zodped.dataset.loader` works against either without a code path for
    "am I reading a snapshot or a working tree".
  * **No raw sensor data.** Samples point at ZOD files by relative path; the bundle ships the
    pointers, never the frames. ZOD is CC BY-SA 4.0 and is the recipient's own download to make.
    This also keeps the bundle at megabytes rather than terabytes.

Every number in the manifest and the summary comes from `zodped.dataset.stats`, computed from the
artifacts being packaged — a snapshot cannot quote a stale count.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import shutil
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd

from zodped.dataset.stats import HORIZONS, funnel_counts, index_summary, pipeline_stages

CHECKSUM_FILE = "CHECKSUMS.sha256"
MANIFEST_FILE = "manifest.json"
SUMMARY_FILE = "README.md"   # the label & tracking summary doubles as the bundle README
ANNOTATIONS_DIR = "annotations"


@dataclass(frozen=True)
class BundleSources:
    """Where the packager reads each part of the snapshot from."""

    annotations: Path        # data/annotations — sample JSONs + dataset_index.parquet
    splits: Path             # Step 4a frozen sequence_splits.json
    schema: Path             # configs/dataset_schema_v0.2.yaml
    trajectories: Path       # Step 1 output, for the funnel counts only (not copied)
    actions: Path            # Step 2 output, for the funnel counts only (not copied)
    actions_verified: Path   # Step 2e output, for the funnel counts only (not copied)
    reports: Path            # per-stage run reports, read for the summary (not copied)
    docs: List[Path]         # reference docs copied in for offline provenance

    def missing(self) -> List[Path]:
        """Required inputs that are not on disk (docs are optional, funnel dirs are not copied)."""
        return [p for p in (self.annotations, self.splits, self.schema) if not p.exists()]


def git_revision(repo_root: Path) -> dict:
    """Current commit and whether the tree was dirty when the bundle was built.

    A dirty tree is recorded, not refused: mid-iteration bundles are legitimate. It just has to be
    visible afterwards, because a dirty bundle is not reproducible from the commit alone.
    """
    def run(*args: str) -> Optional[str]:
        try:
            out = subprocess.run(("git", "-C", str(repo_root)) + args,
                                 capture_output=True, text=True, check=True)
            return out.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file (chunked — sample JSONs are small, Parquet need not be)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(bundle_dir: Path) -> int:
    """Write `CHECKSUMS.sha256` over every file in the bundle. Returns the number of files.

    The checksum file necessarily excludes itself. The manifest is included, so a tampered or
    truncated manifest is detectable.
    """
    target = bundle_dir / CHECKSUM_FILE
    files = sorted(p for p in bundle_dir.rglob("*") if p.is_file() and p != target)
    lines = [f"{sha256(p)}  {p.relative_to(bundle_dir).as_posix()}" for p in files]
    target.write_text("\n".join(lines) + "\n")
    return len(files)


def verify_bundle(bundle_dir: Path) -> List[str]:
    """Re-hash a bundle against its checksum file. Returns a list of problems (empty = intact)."""
    bundle_dir = Path(bundle_dir)
    checksums = bundle_dir / CHECKSUM_FILE
    if not checksums.exists():
        return [f"missing {CHECKSUM_FILE}"]

    problems, listed = [], set()
    for line in checksums.read_text().splitlines():
        if not line.strip():
            continue
        expected, rel = line.split("  ", 1)
        listed.add(rel)
        path = bundle_dir / rel
        if not path.exists():
            problems.append(f"MISSING  {rel}")
        elif sha256(path) != expected:
            problems.append(f"CHANGED  {rel}")

    for path in bundle_dir.rglob("*"):
        if path.is_file() and path != checksums:
            rel = path.relative_to(bundle_dir).as_posix()
            if rel not in listed:
                problems.append(f"UNLISTED {rel}")
    return problems


def build_manifest(sources: BundleSources, index: pd.DataFrame, version: str,
                   repo_root: Path) -> dict:
    """The machine-readable record of what this snapshot is and what is in it."""
    return {
        "name": "zod-ped",
        "status": "INTERNAL SNAPSHOT — not a public release; labels are partially verified",
        "version": version,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "git": git_revision(repo_root),
        "schema": sources.schema.name,
        "description": (
            "Multimodal pedestrian intent & trajectory dataset built on the Zenseact Open Dataset "
            "(ZOD). Camera + LiDAR + radar, where JAAD/PIE/PSI are camera-only."
        ),
        "contents": {
            "annotations": "per-sample JSON + dataset_index.parquet (one row per sample)",
            "splits": "FROZEN sequence-level train/val/test mapping",
            "schema": "authoritative field-by-field spec",
            "docs": "data format, pipeline design, experiments log, JAAD/PIE alignment",
        },
        "raw_sensor_data": {
            "included": False,
            "reason": "samples carry relative pointers; ZOD (CC BY-SA 4.0) is downloaded separately",
            "resolve_with": "zodped.dataset.loader.media_paths(doc, sequences_root)",
        },
        "pipeline_funnel": funnel_counts(sources.trajectories, sources.actions,
                                         sources.actions_verified),
        "pipeline_stages": pipeline_stages(sources.reports),
        "composition": {h: index_summary(index, h) for h in HORIZONS},
        "tier_policy": (
            "Report GOLD and SILVER separately. GOLD is the human-verified evaluation set; SILVER "
            "is weak-labeled training bulk with ~28% measured false-positive rate on declared "
            "crossings. A combined crossing ratio describes neither tier and must not be quoted."
        ),
    }


def _tier_table(summary: dict) -> str:
    """The tier x split composition as a Markdown table, for the summary."""
    header = ("| tier | split | samples | pedestrians | crossers | crossing ratio |\n"
              "|---|---|---:|---:|---:|---:|\n")
    rows = "".join(
        f"| {r['tier']} | {r['split']} | {r['samples']} | {r['pedestrians']} | "
        f"{r['crossers']} | {r['crossing_ratio']:.3f} |\n"
        for r in summary["by_tier_split"]
    )
    return header + rows


def _stage_rows(stages: dict) -> str:
    """The per-stage provenance table: what went in, what came out, what is known to be wrong."""
    step1 = stages.get("step1_trajectories", {})
    action = stages.get("step2_action", {})
    human = stages.get("step2e_human", {})
    samples = stages.get("step3_samples", {}).get("summary", {})
    baseline = stages.get("step4b_baseline", {}).get("gold", {})
    quality = step1.get("quality_tiers") or {}
    rows = []

    if step1:
        n_tracks = (step1.get("gold_tracks") or 0) + (step1.get("silver_tracks") or 0)
        graded = sum(quality.values()) or 1
        rows.append((
            "1 · tracking",
            f"{step1.get('n_sequences')} sequences",
            f"{n_tracks} tracks ({step1.get('gold_tracks')} GOLD + {step1.get('silver_tracks')} SILVER)",
            f"{quality.get('good', 0) / graded:.0%} clean, {quality.get('marginal', 0) / graded:.0%} "
            f"marginal, {quality.get('bad', 0) / graded:.0%} bad" if quality else "—",
        ))
    if action:
        gold_a, silver_a = action.get("gold", {}), action.get("silver", {})
        crossers = gold_a.get("n_crosses_ego_road", 0) + silver_a.get("n_crosses_ego_road", 0)
        rows.append((
            "2 · action (geometry)",
            f"{gold_a.get('n_tracks', 0) + silver_a.get('n_tracks', 0)} pedestrians",
            f"{crossers} declared crossers "
            f"(GOLD {gold_a.get('road_crossing_rate', 0):.1%}, SILVER {silver_a.get('road_crossing_rate', 0):.1%})",
            f"{gold_a.get('n_undetermined', 0)} GOLD tracks undetermined (kept, never forced)",
        ))
    if human:
        reviewed = human.get("human_verified", 0)
        flips = human.get("flipped_cross_to_no", 0) + human.get("flipped_no_to_cross", 0)
        rows.append((
            "2e · human review (GOLD)",
            f"{reviewed} tracks watched",
            f"{human.get('human_cross', 0)} crossing / {human.get('human_no_cross', 0)} not",
            f"**{flips} labels flipped** ({human.get('flipped_cross_to_no', 0)} cross→no, "
            f"{human.get('flipped_no_to_cross', 0)} no→cross) = geometry's measured error",
        ))
    if samples:
        rows.append((
            "3 · intent windows",
            f"{samples.get('n_pedestrians')} pedestrians",
            f"{samples.get('n_samples')} samples ({samples.get('n_tte_anchored')} TTE-anchored, "
            f"{samples.get('n_closest_approach')} comparison)",
            "labels forward-looking; filters applied per window",
        ))
    if baseline:
        rows.append((
            "4b · sanity check",
            f"{baseline.get('n_train_windows')} train windows",
            f"GOLD test AUC **{baseline.get('gold_test_auc_mean')}** ± {baseline.get('gold_test_auc_std')}",
            f"AP {baseline.get('gold_test_ap_mean')} vs chance {baseline.get('positive_rate')} — labels learnable",
        ))

    header = ("| stage | in | out | what we know about the error |\n|---|---|---|---|\n")
    return header + "".join(f"| {a} | {b} | {c} | {d} |\n" for a, b, c, d in rows)


def _skip_rows(skip_counts: dict) -> str:
    """Step-3 window rejections, largest first — where the pedestrians that did not ship went."""
    if not skip_counts:
        return "_(no skip counts in the run report)_\n"
    total = sum(skip_counts.values())
    rows = "".join(
        f"| `{reason}` | {count} | {count / total:.0%} |\n"
        for reason, count in sorted(skip_counts.items(), key=lambda kv: -kv[1])
    )
    return (f"| gate | windows dropped | share |\n|---|---:|---:|\n{rows}"
            f"| **total** | **{total}** | |\n")


def build_summary(manifest: dict, horizon: str) -> str:
    """Generate the label & tracking summary — the snapshot's README and the report backbone.

    Every number is filled from the manifest and the run reports, so the document cannot drift
    from the artifacts it describes. Prose that needs a human is marked TODO rather than invented.

    This is written as an INTERNAL document. The labels are not release-grade (see the honest
    accounting below), so it describes what was done and how wrong it is known to be, rather than
    inviting anyone to build on it.
    """
    summary = manifest["composition"][horizon]
    gold = summary["by_tier"].get("GOLD", {})
    silver = summary["by_tier"].get("SILVER", {})
    totals = summary["totals"]
    funnel = manifest["pipeline_funnel"]
    stages = manifest.get("pipeline_stages", {})
    human = stages.get("step2e_human", {})
    step3 = stages.get("step3_samples", {})
    git = manifest["git"]
    commit = (git.get("commit") or "unknown")[:12]
    dirty = " (tree dirty at build time)" if git.get("dirty") else ""
    flips = human.get("flipped_cross_to_no", 0) + human.get("flipped_no_to_cross", 0)
    reviewed = human.get("human_verified", 0) or 1
    gold_test = next((r for r in summary["by_tier_split"]
                      if r["tier"] == "GOLD" and r["split"] == "test"), {})

    return f"""# zod-ped — Label & Tracking Summary  (snapshot v{manifest["version"]})

> **INTERNAL — NOT FOR DISTRIBUTION.** The labels are auto-generated and only partially
> human-verified. This snapshot exists to pin a set of numbers to one exact state of the data, so
> results stay reproducible while the work continues. It is not a public dataset release, and the
> limitations below are the reason.

{manifest["description"]}

Built {manifest["created_utc"]} from commit `{commit}`{dirty}.
Schema: `schema/{manifest["schema"]}`.

## Read this first — what state the labels are actually in

1. **Only {human.get("human_verified", 0)} tracks of {funnel["step2_pedestrians"]["total"]} have been watched by a human.** Everything else carries a geometric rule's verdict.
2. **On those reviewed tracks, human review flipped {flips} labels ({flips / reviewed:.0%})** — {human.get("flipped_cross_to_no", 0)} declared crossings that were not, and {human.get("flipped_no_to_cross", 0)} crossings the rule missed. That is the measured error rate of the automatic labels, and SILVER carries it entirely.
3. **{human.get("pv_disputed_unreviewed", 0)} GOLD tracks flagged as disputed were never reviewed.**
4. **SILVER's accuracy has never been measured directly** — too few SILVER tracks carry a human label. Its error rate is GOLD's, carried over on the assumption the rule behaves the same; SILVER's tracks are farther and noisier, so that assumption is optimistic.
5. **The evaluation set is small** — GOLD test holds {gold_test.get("crossers", 0)} positive windows out of {gold_test.get("samples", 0)}. Differences of a few points between models mean nothing at that size.
6. **The labels are contested even among humans** — two reviewers agreed on ~86% of the verified tracks, so this task has no clean ceiling.

Nothing here is a surprise or a regression; it is the honest accounting, and it is why this is a
snapshot rather than a release.

## What was done to the data

{_stage_rows(stages)}
Design rationale for each stage → `docs/PIPELINE.md`. Dated evidence and the rejected
alternatives → `docs/EXPERIMENTS_LOG.md`. Detector and frustum bring-up numbers live only in the
log, deliberately, so they have exactly one home.

## What a sample is

One sample is a **(pedestrian, time window)** pair: a 0.5s observation window over one tracked
pedestrian, carrying a forward-looking intent label.

**Action is not intent.** `action.crosses_ego_road` is a verdict about the *whole track* — did
this person cross the ego road, and when. `intent.labels_by_horizon` is *per window* and looks
*forward* — will crossing start within the horizon **after** the window ends. Training on the
action field turns intent prediction into action detection and breaks comparability with JAAD/PIE.

Labels are provided at three prediction horizons: {", ".join(HORIZONS)} seconds.

## Contents

```
annotations/     {totals["samples"]} sample JSONs + dataset_index.parquet (one row per sample)
splits/          FROZEN sequence-level train/val/test mapping
schema/          authoritative field-by-field spec
docs/            data format, pipeline design, experiments log, JAAD/PIE alignment
manifest.json    build provenance + full composition at every horizon
{CHECKSUM_FILE}  SHA-256 of every file above
```

**Raw sensor data is not included.** Samples point at ZOD camera/LiDAR/radar files by path
relative to the sequence directory. Download ZOD separately and resolve them with
`zodped.dataset.loader.media_paths`.

## Composition (horizon {horizon}s)

{totals["samples"]} samples over {totals["pedestrians"]} pedestrians in {totals["sequences"]} sequences.

{_tier_table(summary)}
### Read this before quoting a number

**GOLD and SILVER are reported separately, always.**

| tier | what it is | samples | crossing ratio |
|---|---|---:|---:|
| **GOLD** | human-verified labels — **the evaluation set** | {gold.get("samples", 0)} | **{gold.get("crossing_ratio", 0):.3f}** |
| **SILVER** | geometry labels, weak — training bulk, never evaluation | {silver.get("samples", 0)} | {silver.get("crossing_ratio", 0):.3f} |

SILVER's crossing rate is far lower by construction, not by error: ZOD annotates only pedestrians
a human judged relevant, and relevant overwhelmingly means near the road, while the detector finds
everyone else — far, peripheral, on the pavement. A combined ratio therefore describes neither
tier. Every index row carries `is_in_gold_standard`, so the two are always separable.

## Loading

```python
from zodped.dataset.loader import (load_index, select, load_sample,
                                   boxes_array, positions_array, intent_label)

index = load_index("annotations")
train = select(index, split="train", tier="gold")

doc = load_sample(train.sample_id.iloc[0], "annotations")
boxes = boxes_array(doc)                       # (T, 4) pixel xyxy, NaN where unobserved
past  = positions_array(doc)                   # (T, 3) world xyz INSIDE the window  -> input
future = positions_array(doc, part="future")   # after the window                    -> target
label = intent_label(doc, horizon="{horizon}")

# Raw sensors, if you have ZOD on disk:
from zodped.dataset.loader import media_paths, load_radar_window
paths = media_paths(doc, "data/raw/sequences")
radar = load_radar_window(doc, "data/raw/sequences")   # returns inside this window
```

> **`trajectory.frames` is not the observation window.** It spans
> `[window_start, window_end + max_horizon]`, so most of its rows lie *after* the window — they
> are the trajectory-prediction target. Passing them to a model as input leaks the answer.
> `positions_array` defaults to `part="observed"` for exactly this reason; the raw JSON gives you
> no such protection. (The per-frame `in_observation` flag is unrelated — it means the tracker had
> a real detection rather than coasting.) `boxes_array` is window-only and always safe.

Splits are **sequence-level and frozen** — windows from one sequence share frames, pedestrians and
scene, so a sample-level split would leak near-duplicates into test. Use the shipped `split`
column; do not re-deal it.

## Where the pedestrians went

Funnel: {funnel["step1_trajectory_files"]["total"]} track files →
{funnel["step2_pedestrians"]["total"]} pedestrians after cut + stitch →
{totals["samples"]} samples. Step 3 dropped the rest through explicit per-window gates:

{_skip_rows(step3.get("skip_counts", {}))}
The `distance_to_ego` gate is the largest single filter and is deliberate — it is a class-balance
filter, not a data-availability one. Relaxing it makes the crossing ratio *worse*, because the
pedestrians it admits are far ones who never cross. Measured; see EXPERIMENTS_LOG.

## Other caveats

Beyond the label-quality accounting at the top:

1. **Geometry is FOV- and range-limited by construction.** The `ego_road` polygon exists only at
   the keyframe camera, so crossings outside that view cannot be seen by the rule at all.
2. **Radar is per-sequence, not per-window.** ZOD ships one structured `.npy` per sequence holding
   every sweep; `radar_path` points at that blob. Use `loader.load_radar_window` to slice it.
3. **`num_pedestrians_in_scene` / `is_key_pedestrian` are population-dependent.** The values here
   come from the full-population pass and are the accurate ones, but re-running the pipeline over
   a single tier would recompute them differently. Known wart, not yet fixed.
4. **The model committee was tried and dropped.** PV-LSTM ranks review candidates well but decides
   badly (0.29 precision against geometry's 0.72), so it labels nothing. Evidence in
   EXPERIMENTS_LOG.

## Provenance and licence

Underlying sensor data is the **Zenseact Open Dataset**, CC BY-SA 4.0, obtained separately from
<https://zod.zenseact.com/> under its own terms. This snapshot redistributes no ZOD data — only
annotations and relative pointers. Code: MIT.

Produced at the Intelligent Robotics Lab, Aalto University.
"""


@dataclass(frozen=True)
class BundleResult:
    """What a build produced: where it went, its manifest, and how many files it checksummed.

    `n_files` is returned rather than stored in the manifest, because the checksum pass necessarily
    runs after `manifest.json` is written — a count inside the manifest would always be the count
    from before the manifest itself existed.
    """

    path: Path
    manifest: dict
    n_files: int


def build_bundle(sources: BundleSources, out_dir: Path, version: str, repo_root: Path,
                 horizon: str, overwrite: bool = False) -> BundleResult:
    """Assemble the snapshot bundle at `out_dir`.

    Refuses to write into an existing directory unless `overwrite` — a snapshot is something
    results may already have been reported against, and silently merging a new build into an old
    one produces a bundle whose checksums pass but whose contents are from two runs.
    """
    missing = sources.missing()
    if missing:
        raise FileNotFoundError("missing required inputs: " + ", ".join(str(p) for p in missing))

    out_dir = Path(out_dir)
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{out_dir} exists — pass --overwrite to replace it")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    shutil.copytree(sources.annotations, out_dir / ANNOTATIONS_DIR)
    (out_dir / "splits").mkdir()
    shutil.copy2(sources.splits, out_dir / "splits" / sources.splits.name)
    (out_dir / "schema").mkdir()
    shutil.copy2(sources.schema, out_dir / "schema" / sources.schema.name)
    if sources.docs:
        (out_dir / "docs").mkdir()
        for doc in sources.docs:
            if doc.exists():
                shutil.copy2(doc, out_dir / "docs" / doc.name)

    index = pd.read_parquet(out_dir / ANNOTATIONS_DIR / "dataset_index.parquet")
    manifest = build_manifest(sources, index, version, repo_root)
    (out_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2) + "\n")
    (out_dir / SUMMARY_FILE).write_text(build_summary(manifest, horizon))

    return BundleResult(path=out_dir, manifest=manifest, n_files=write_checksums(out_dir))


def make_tarball(bundle_dir: Path, out_path: Optional[Path] = None) -> Path:
    """Compress a built bundle to `<bundle>.tar.gz`, with the bundle directory as the tar root."""
    bundle_dir = Path(bundle_dir)
    # Name-append, not with_suffix: a version like "v0.2" makes ".2" look like a suffix.
    out_path = Path(out_path) if out_path else bundle_dir.parent / f"{bundle_dir.name}.tar.gz"
    with tarfile.open(out_path, "w:gz") as tar:
        tar.add(bundle_dir, arcname=bundle_dir.name)
    return out_path
