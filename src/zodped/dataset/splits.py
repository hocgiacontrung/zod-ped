"""Sequence-level train/val/test split assignment (Step 4).

The split unit is the SEQUENCE, not the sample: windows of one sequence share frames,
pedestrians, and scene (street, lighting, ego drive), so sample-level splitting would leak
near-duplicates of a training window into the test set and inflate every benchmark number.
Assigning whole sequences keeps the test set genuinely unseen.

The assignment is stratified on the scarce resource — TTE-anchored crosser windows
(242 over 1,087 GOLD samples) — so a random deal cannot starve one split of positives.
Sequences are bucketed by crosser-window count (2+, 1, 0) and dealt greedily within each
bucket to the split furthest below its target ratio, crosser mass first, sample mass as
tie-break. Zero-sample sequences of the working set are dealt too (weight 1), so a later
SILVER pour inherits a split instead of needing a new deal.

The resulting {seq_id: split} mapping is written once and treated as FROZEN: labels will be
upgraded (model-consensus action, behavioral intent) and re-poured through Steps 2-4, and
results across label versions stay comparable only if they share the same test sequences.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")
DEFAULT_RATIOS = (0.7, 0.1, 0.2)


@dataclass(frozen=True)
class SequenceStats:
    """Per-sequence sample mass, as seen by the stratified deal."""

    seq_id: str
    n_samples: int
    n_crosser_windows: int   # TTE-anchored windows = the scarce positive class


def stats_from_index(index: pd.DataFrame, all_seq_ids: Sequence[str]) -> List[SequenceStats]:
    """Per-sequence stats from the Step-3 Parquet index, covering ALL working-set sequences.

    `all_seq_ids` is the full pedestrian working set; sequences without samples yet still get
    stats (zeros) so they receive a split now rather than at some future re-deal.
    """
    grouped = index.groupby("sequence_id")
    n_samples = grouped.size().to_dict()
    n_crossers = (
        index[index["window_kind"] == "tte_anchored"].groupby("sequence_id").size().to_dict()
    )
    seq_ids = sorted(set(all_seq_ids) | set(n_samples))
    return [
        SequenceStats(s, int(n_samples.get(s, 0)), int(n_crossers.get(s, 0)))
        for s in seq_ids
    ]


def assign_splits(
    stats: List[SequenceStats],
    ratios: Sequence[float] = DEFAULT_RATIOS,
    seed: int = 42,
) -> Dict[str, str]:
    """Deal sequences into train/val/test, stratified on crosser-window count.

    Greedy weighted deal: strata are processed scarcest-first (2+ crossers, 1, 0); inside a
    stratum the order is seed-shuffled, and each sequence goes to the split with the lowest
    projected fill relative to its target ratio — crosser mass is the primary fill metric,
    sample mass the tie-break (and the sole metric for the crosser-free stratum, where
    zero-sample sequences count as weight 1 so they spread too).
    """
    if len(ratios) != len(SPLITS) or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must be {len(SPLITS)} values summing to 1, got {ratios}")
    target = dict(zip(SPLITS, ratios))

    rng = np.random.default_rng(seed)
    strata: Dict[int, List[SequenceStats]] = {2: [], 1: [], 0: []}
    for st in stats:
        strata[min(st.n_crosser_windows, 2)].append(st)

    # Both fill metrics are normalized by their corpus totals so they are commensurate:
    # fill(s) = 1.0 means split s already holds its full target share of that resource.
    total_cross = max(sum(st.n_crosser_windows for st in stats), 1)
    total_samp = max(sum(max(st.n_samples, 1) for st in stats), 1)

    acc_cross = {s: 0.0 for s in SPLITS}
    acc_samp = {s: 0.0 for s in SPLITS}
    assignments: Dict[str, str] = {}
    for bucket in (2, 1, 0):                        # scarcest first
        members = sorted(strata[bucket], key=lambda st: st.seq_id)
        rng.shuffle(members)
        for st in members:
            w_cross = float(st.n_crosser_windows)
            w_samp = float(max(st.n_samples, 1))    # zero-sample seqs still carry weight

            def fill(s: str) -> float:
                """Projected fill of split s if it took this sequence: crosser mass dominant
                (weight 1.0), sample mass secondary (0.25) — enough to steer sample balance
                without letting a big crosser-free scene outvote a scarce positive. In the
                crosser-free stratum w_cross is 0 and the crosser fills are already ~equal,
                so the sample term decides — repairing the skew the crosser strata left."""
                cross_fill = (acc_cross[s] + w_cross) / (target[s] * total_cross)
                samp_fill = (acc_samp[s] + w_samp) / (target[s] * total_samp)
                return cross_fill + 0.25 * samp_fill

            split = min(SPLITS, key=fill)
            assignments[st.seq_id] = split
            acc_cross[split] += w_cross
            acc_samp[split] += w_samp
    return assignments


def balance_score(stats: List[SequenceStats], assignments: Dict[str, str],
                  ratios: Sequence[float] = DEFAULT_RATIOS) -> float:
    """L1 distance of a deal from its target fractions (crosser + sample mass); 0 = perfect.

    Sequences are lumpy (one scene can carry 30+ windows), so any single greedy deal lands a
    few percent off target by luck of the shuffle. Scoring lets the caller run several seeds
    and freeze the best deal instead of the first one.
    """
    target = dict(zip(SPLITS, ratios))
    by_split = {s: [st for st in stats if assignments[st.seq_id] == s] for s in SPLITS}
    total_cross = max(sum(st.n_crosser_windows for st in stats), 1)
    total_samp = max(sum(st.n_samples for st in stats), 1)
    return sum(
        abs(sum(st.n_crosser_windows for st in by_split[s]) / total_cross - target[s])
        + abs(sum(st.n_samples for st in by_split[s]) / total_samp - target[s])
        for s in SPLITS
    )


def split_summary(index: pd.DataFrame) -> dict:
    """Achieved per-split composition (expects the index's `split` column to be filled)."""
    horizons = sorted(c for c in index.columns if c.startswith("intent_h"))
    out: Dict[str, dict] = {}
    for split in SPLITS:
        part = index[index["split"] == split]
        n = len(part)
        out[split] = {
            "n_sequences": int(part["sequence_id"].nunique()),
            "n_samples": n,
            "n_tte_anchored": int((part["window_kind"] == "tte_anchored").sum()),
            "sample_fraction": round(n / len(index), 4) if len(index) else None,
            "per_horizon": {
                h: {
                    "crossing": int((part[h] == "crossing").sum()),
                    "crossing_ratio": round((part[h] == "crossing").mean(), 4) if n else None,
                }
                for h in horizons
            },
        }
    return out
