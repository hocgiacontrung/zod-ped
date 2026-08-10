"""The dataset's numbers as DATA — computed once, consumed by several callers.

`scripts/dataset_stats.py` prints these; `zodped.dataset.packaging` embeds them in the release
manifest and dataset card. Neither re-derives a count, so the console and the shipped bundle can
never disagree about how many samples there are.

Every number here is read off the artifacts on disk. Nothing is quoted from a doc.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

HORIZONS = ("1.0", "1.5", "2.0")
DEFAULT_HORIZON = "2.0"
TIERS = ("GOLD", "SILVER")


def horizon_column(horizon: str = DEFAULT_HORIZON) -> str:
    """Index column holding the intent label at `horizon` seconds (``2.0`` -> ``intent_h2_0s``)."""
    if horizon not in HORIZONS:
        raise ValueError(f"horizon must be one of {HORIZONS}, got {horizon!r}")
    return f"intent_h{horizon.replace('.', '_')}s"


def tier_counts(paths: Iterable[Path], key: str = "is_in_gold_standard") -> Counter:
    """GOLD/SILVER split of a directory of pipeline docs, read from the tier flag on each."""
    counts: Counter = Counter()
    for path in paths:
        counts["gold" if json.loads(path.read_text()).get(key, True) else "silver"] += 1
    return counts


def funnel_counts(traj_dir: Path, actions_dir: Path, verified_dir: Path) -> dict:
    """Pipeline funnel: Step-1 track FILES -> Step-2 pedestrians -> Step-2e labeled records.

    The Step-1 count is files and the Step-2 count is pedestrians; they differ because the cut +
    stitch pass folds SILVER fragments into one walker (see `zodped.labeling.stitching`). Keeping
    both is the point of the funnel — it shows where the drop happened.
    """
    traj = tier_counts(p for p in traj_dir.glob("*.json") if not p.name.startswith("_"))
    actions = tier_counts(actions_dir.glob("*.json"))
    verified = list(verified_dir.glob("*.json"))
    human = sum(bool(json.loads(p.read_text()).get("human_review")) for p in verified)
    return {
        "step1_trajectory_files": {"total": sum(traj.values()), **dict(traj)},
        "step2_pedestrians": {"total": sum(actions.values()), **dict(actions)},
        "step2e_labeled_records": {"total": len(verified), "human_verified": human},
    }


def _read(path: Path) -> Optional[dict]:
    """Load a run report, or None if that stage has not been run / its report was cleaned away."""
    return json.loads(path.read_text()) if path.exists() else None


def pipeline_stages(reports_dir: Path) -> dict:
    """What each pipeline stage did to the data, read from the run reports it wrote.

    This is the provenance spine of the label summary: for every stage, how many units went in,
    how many came out, and what is known about the error it introduces. Stages whose report is
    absent are omitted rather than guessed at.

    Two measurements deliberately live only in `docs/EXPERIMENTS_LOG.md` and are NOT restated
    here: the Step-0 detector bring-up gates and the Step-1 frustum localisation error. They come
    from one-off evaluations against ZOD ground truth, not from a pipeline run, so no artifact on
    disk carries them and copying them into a generated file would create a second source of truth.
    """
    reports_dir = Path(reports_dir)
    traj, silver = _read(reports_dir / "trajectories_run_report.json"), _read(reports_dir / "silver_run_report.json")
    quality = _read(reports_dir / "trajectory_quality_report.json")
    act_gold, act_silver = _read(reports_dir / "actions_run_report.json"), _read(reports_dir / "actions_silver_run_report.json")
    human = _read(reports_dir / "human_merge_report.json")
    samples = _read(reports_dir / "samples_run_report.json")
    splits = _read(reports_dir / "splits_report.json")
    baselines = {arm: _read(reports_dir / f"baseline_pvlstm_{arm}.json") for arm in ("gold", "all")}

    stages: dict = {}
    if traj or silver:
        stages["step1_trajectories"] = {
            "n_sequences": (traj or silver).get("n_sequences"),
            "gold_tracks": (traj or {}).get("n_trajectories"),
            "silver_tracks": (silver or {}).get("n_silver_tracks"),
            "measurement": (traj or {}).get("config", {}).get("detector"),
            "quality_tiers": (quality or {}).get("tier_counts"),
            "quality_rule": (quality or {}).get("tier_rule"),
        }
    if act_gold or act_silver:
        stages["step2_action"] = {
            tier: report["summary"]
            for tier, report in (("gold", act_gold), ("silver", act_silver)) if report
        }
    if human:
        stages["step2e_human"] = human["counts"]
    if samples:
        stages["step3_samples"] = {"config": samples.get("config"),
                                   "summary": samples.get("summary"),
                                   "skip_counts": samples.get("skip_counts")}
    if splits:
        stages["step4a_splits"] = {"config": splits.get("config"), "summary": splits.get("summary")}
    arms = {arm: report["summary"] for arm, report in baselines.items() if report}
    if arms:
        stages["step4b_baseline"] = arms
    return stages


def _annotate(index: pd.DataFrame, horizon: str) -> pd.DataFrame:
    """Index plus the three derived columns every summary needs.

    `ped_key` is sequence-scoped because SILVER pedestrian ids (``silver015``) are only unique
    within their sequence — counting `pedestrian_id` alone would collapse thousands of distinct
    people into one.
    """
    df = index.copy()
    df["tier"] = df.is_in_gold_standard.map({True: "GOLD", False: "SILVER"})
    df["ped_key"] = df.sequence_id + "_" + df.pedestrian_id
    df["crossing"] = df[horizon_column(horizon)] == "crossing"
    return df


def summary_table(index: pd.DataFrame, horizon: str = DEFAULT_HORIZON) -> pd.DataFrame:
    """The tier x split composition table — the one groupby both consumers share."""
    df = _annotate(index, horizon)
    table = df.groupby(["tier", "split"]).agg(
        samples=("crossing", "size"),
        peds=("ped_key", "nunique"),
        crossers=("crossing", "sum"),
    )
    table["ratio"] = (table.crossers / table.samples).round(3)
    return table


def index_summary(index: pd.DataFrame, horizon: str = DEFAULT_HORIZON) -> dict:
    """Composition of the shipped index at one prediction horizon, as a JSON-safe dict.

    The per-tier blocks are deliberately the headline. GOLD is the human-verified evaluation set
    and SILVER is weak-labeled training bulk, so a combined crossing ratio describes neither tier
    (GOLD runs ~6.6x higher); this function therefore reports no corpus-wide ratio.
    """
    df = _annotate(index, horizon)
    table = summary_table(index, horizon)

    by_tier = {}
    for tier in TIERS:
        part = df[df.tier == tier]
        if part.empty:
            continue
        by_tier[tier] = {
            "samples": int(len(part)),
            "pedestrians": int(part.ped_key.nunique()),
            "sequences": int(part.sequence_id.nunique()),
            "crossers": int(part.crossing.sum()),
            "crossing_ratio": round(float(part.crossing.mean()), 4),
        }

    return {
        "horizon_s": horizon,
        "totals": {
            "samples": int(len(df)),
            "pedestrians": int(df.ped_key.nunique()),
            "sequences": int(df.sequence_id.nunique()),
        },
        "by_tier": by_tier,
        "by_tier_split": [
            {
                "tier": tier,
                "split": split,
                "samples": int(row.samples),
                "pedestrians": int(row.peds),
                "crossers": int(row.crossers),
                "crossing_ratio": float(row.ratio),
            }
            for (tier, split), row in table.iterrows()
        ],
    }
