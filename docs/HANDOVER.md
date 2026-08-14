# Handover — how to run this

Short operating guide. Design lives in `PIPELINE.md`, evidence in `EXPERIMENTS_LOG.md`, current
state in `CLAUDE.md`.

## Read in this order

1. `README.md` — what the dataset is, how to install.
2. `CLAUDE.md` — current state of every step, and the standing rules.
3. `docs/LABEL_SUMMARY.md` — what each stage did to the data and what is known to be wrong with it.
4. `docs/PIPELINE.md` — why the pipeline is shaped this way.
5. `configs/dataset_schema_v0.2.yaml` — field-by-field spec, when you need a specific field.

`docs/DATA_FORMAT.md` and `docs/EXPERIMENTS_LOG.md` are references — look things up, don't read
front to back.

## Setup

```bash
conda activate zod-iac
pip install -r requirements.txt
pip install -e . --no-deps
```

ZOD is downloaded separately (see `docs/DATA_FORMAT.md`).

## Just using the data

You do not need to re-run anything. The shipped samples are in `data/annotations/`.

```python
from zodped.dataset.loader import load_index, select, load_sample, positions_array, intent_label

index = load_index("data/annotations")
train = select(index, split="train", tier="gold")   # evaluate on GOLD only

doc   = load_sample(train.sample_id.iloc[0], "data/annotations")
past  = positions_array(doc)                        # window only — safe model input
label = intent_label(doc, horizon="2.0")
```

Current numbers: `python scripts/dataset_stats.py`.

## Re-running the pipeline

Run from the repo root, in order. Steps 0–1 are the expensive ones (GPU, hours); 2–4 are minutes.

```bash
python scripts/00_detect.py                                  # 2D detection cache, all 358 seqs
python scripts/01_generate_trajectories.py                   # 1a GOLD tracks
python scripts/01b_generate_silver.py                        # 1b SILVER tracks

python scripts/silver_cut.py                                 # SILVER QC: junk cut  -> review/silver_cut.json
python scripts/stitch_tracks.py --base-residual-m 0.6 --residual-per-s 0.2
                                                             # SILVER QC: fragment merge -> review/stitch_proposals_final.json

python scripts/02_label_action.py  --tier all --stitch       # geometry action labels
python scripts/02e_merge_human_labels.py                     # overlay human verdicts (GOLD)

python scripts/03_assemble_samples.py --tier all --stitch \
       --actions-dir data/processed/actions_verified         # samples + intent labels

python scripts/04b_train_baseline.py --train-tiers gold      # label sanity check
python scripts/05_package_snapshot.py --overwrite --summary-to docs/LABEL_SUMMARY.md
```

Three flags decide whether you reproduce the shipped numbers or something else:

- **`--actions-dir data/processed/actions_verified`** on Step 3. It defaults to `actions/`, which is
  geometry-only — omit it and you silently discard the human review pass. Counts stay plausible, so
  nothing warns you.
- **`--tier all`** on Steps 2 and 3. Defaults to `gold`. A single-tier run also recomputes
  `num_pedestrians_in_scene` / `is_key_pedestrian` differently (known wart, see `CLAUDE.md`).
- **`--stitch`** on Steps 2 and 3, so SILVER is labeled per pedestrian rather than per fragment.

**Do not re-run `scripts/04_assign_splits.py`.** The splits are FROZEN and inherited by everything
reported so far; re-dealing them invalidates every number in the docs. It refuses without `--force`.

## Where things live

Nothing under `data/` is in git. ★ marks what a re-run cannot rebuild — back these up before
touching the machine they live on.

| path | what |
|---|---|
| `src/zodped/` | the library — all real logic (in git) |
| `scripts/` | thin entry-points, argparse + I/O (in git) |
| ★ `data/processed/review/` | human review worksheets — hand-made |
| ★ `data/processed/actions_verified/` | the merged human layer Step 3 reads |
| ★ `data/processed/splits/` | the FROZEN split mapping |
| ★ `data/annotations/` | shipped samples + `dataset_index.parquet` |
| `data/processed/{detections,trajectories,actions}/` | rebuild from ZOD via the commands above |
| `data/snapshots/` | rebuild with Step 4c |

```bash
tar czf zod-ped-human-labels.tar.gz \
    data/processed/review data/processed/actions_verified \
    data/processed/splits data/annotations          # ~130 MB
```

## Adding to the pipeline

Logic goes in `src/zodped/`, not in `scripts/` — scripts are argparse + I/O and call into the
library. Shared paths and the `--tier` / frustum flags come from `scripts/_common.py`. Anything
that reports a count should go through `zodped.dataset.stats`, so there stays exactly one counting
path.
