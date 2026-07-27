# JAAD / PIE Alignment — Discussion Draft

> **STATUS: DRAFT FOR DEBATE (2026-07-02). Nothing here is adopted.** No schema change has
> been made. This doc catalogues what JAAD and PIE actually annotate, maps each field onto our
> pipeline, and frames the open decisions. Treat every "Proposal" below as a candidate to argue
> about with the supervisor, not a settled design. Schema edits happen only *after* those calls.
>
> **Update 2026-07-15:** one adjacent decision IS now settled (in PIPELINE.md / the schema): 
the **ACTION** (track-level verdict) label source = **model consensus** (JAAD/PIE-trained
> models), validated against **human curated labels** (geometry retired as the yardstick 2026-07-27,
> ~30% wrong vs a human — §5). Per-window **intent** stays
> a separate later step. The taxonomy/naming/benchmark-shape questions below remain open — and §2.4
> (benchmark-interface alignment) now also bears on how we feed windows into those three models.

## Why this doc exists

JAAD and PIE are the two established camera-only pedestrian-intent datasets — years of labeling
decisions we'd otherwise rediscover the hard way. Our novelty is the extra modalities (camera +
LiDAR + radar) and auto-labeling at scale, but the *label taxonomy* is a solved problem worth
borrowing, for quality and benchmark comparability. This doc is that reference; specs below come
from the JAAD/PIE repos/pages (verified 2026-07-02), not memory (see "Sources").

---

## 1. Verified annotation specs (evidence base)

### JAAD — agent classes
Three bounding-box categories, distinguished by **relevance + resolvability** (NOT by data quality):

| class | id suffix | annotated? | meaning |
|-------|-----------|------------|---------|
| `pedestrian` | `b` (e.g. `0_1_3b`) | full behavior + attributes | near the road, interacts with the ego |
| `ped` | — | **bbox only** | bystander, far, no ego interaction |
| `people` | `p` (e.g. `0_5_2p`) | single group box | a group, not individually resolved |

`sample_type='all'` vs `'beh'` selects all pedestrians vs only the behaviorally-annotated subset.

### PIE — behavioral annotations (per frame, per pedestrian)
- `action`: `walking` | `standing`
- `gesture`: `hand_ack` | `hand_yield` | `hand_rightofway` | `nod` | `other`
- `look`: `looking` | `not-looking` (at the ego vehicle)
- `cross`: `not-crossing` | `crossing` | `crossing-irrelevant`

### PIE — pedestrian attributes
- `age`: `child` | `adult` | `senior`; `gender`: `male` | `female`
- `num_lanes` (scalar); `signalized`: `n/a` | `C` | `S` | `CS`
- `traffic_direction`: `OW` | `TW`; `intersection`: `midblock` | `T` | `T-right` | `T-left` | `four-way`
- `crossing`: `1` | `0` | `-1` (crossing / not / irrelevant)
- `exp_start_point`, `critical_point`: first/last frame shown to human raters
- `intention_prob`: `[0,1]`, aggregated from a **human experiment**
- `crossing_point`: frame where the pedestrian **starts** crossing

### Occlusion (both)
- Pedestrians: `0` none, `1` partial (25–75%), `2` full (>75%).
- Other objects: `0` visible, `1` partial/full.

### PIE — ego-vehicle / OBD (per frame)
`GPS_speed` (km/h), `OBD_speed` (km/h), `heading_angle`, `latitude`, `longitude`, `pitch`, `roll`,
`yaw`, `acceleration`, `gyroscope`.

### JAAD — appearance attributes (24-dim binary, manual)
`pose_{front,back,left,right}`, `clothes_below_knee`, `clothes_upper_{light,dark}`,
`clothes_lower_{light,dark}`, `backpack`, `bag_{hand,arm,shoulder,left_side,right_side}`, `cap`,
`hood`, `sunglasses`, `umbrella`, `phone`, `baby`, `object`, `stroller/cart`, `bicycle/motorcycle`.

---

## 2. The findings that actually change our design

### 2.1 `agent_class` (pedestrian / ped / people) is orthogonal to GOLD/SILVER — we likely want both
GOLD/SILVER is a **provenance** axis (how a track is *born*: keyframe-anchored vs detector-birth).
JAAD's class is a **relevance/resolvability** axis. They are independent, and mixing them up would be
a mistake.

Mapping onto us:

| JAAD class | our equivalent |
|------------|----------------|
| `pedestrian` | `determined` track near corridor/road → becomes an intent sample |
| `ped` | `determined`-but-far track **+ our EMPTY/`undetermined` stubs** → kept as detections, excluded from intent samples (this is already our "keep + flag" behaviour) |
| `people` | **our crowd identity-swap failure mode** (see EXPERIMENTS_LOG: 2% speed-kick swaps, concentrated in the seq-000603 34-ped crowd) |

**Why the `people` row matters most.** JAAD's answer to dense crowds is *don't individually track
them* — emit one group box, no per-person behavior. That is a principled escape hatch for exactly the
regime where our tracker is weakest. It means we do **not** have to solve crowd ReID to ship a clean
dataset: we demote crowds to group boxes, honestly, and exclude them from per-pedestrian labels.

**Proposal (to debate):** add an `agent_class ∈ {pedestrian, ped, people}` concept, kept separate from
`is_in_gold_standard`. **Open questions:** (a) what geometric rule promotes a track to `pedestrian`
(near corridor? near road? within Nm?) vs demotes to `ped`? (b) what density/overlap threshold triggers
`people`, and do we detect it from box overlap, from tracker fragmentation, or from detection density?
(c) does `people` get a corridor/road action at the group level, or no action at all?

### 2.2 Action vs intent: our split is right, but "intent" is the wrong *name* for what Step 3 computes
PIE keeps these as **distinct** fields: `cross` (per-frame action), `crossing_point` (≈ our `t_c`), and
`intention_prob` — the last one gathered from a **human experiment** (subjects watch up to
`critical_point`, rate "will they cross?").

The uncomfortable part: **`intention_prob` ≠ "does the pedestrian cross within the horizon".** Intention
is a *judgement about a person's state*; a pedestrian can intend to cross and then not (hesitation,
gap rejection), or drift across without prior intent. What our Step 3 derives geometrically —
`t_c ∈ [window_end, window_end+h]` — is **future action within a horizon**, which PIE would call the
*crossing-prediction* label, not `intention_prob`.

This is fine and standard (SF-GRU / PCPA benchmarks predict the `cross` action, not `intention_prob`),
and we have no budget for a human experiment. But we should **name it honestly**:

**Proposal (to debate):** call the Step-3 output a *crossing-prediction* label. Reserve the word
"intention" for the PIE sense (a forward-looking judgement on the observation window alone), which we
could only approximate later via pose ("looking") or a VLM on the edge-case budget. **Open question:**
do we rename now, keep "intent" as a loose alias, or attempt a cheap intention proxy at all?

### 2.3 Class imbalance — feeds directly into the corridor-vs-union decision
JAAD/PIE are heavily crossing-imbalanced, so JAAD ships both `all` and `beh` (behavioral subset)
views. Our corridor-alone rate (**2.4%** of determined) is *more* extreme than theirs; the broader
`ego_road` rate (**9.1%**) lands much closer to JAAD/PIE crossing ratios.

**Implication:** this is an *empirical* argument in the separate corridor-vs-union debate — union isn't
just conceptually broader, it also produces a benchmark-usable positive rate. And regardless of that
outcome, we should mirror JAAD's `all` + behavioral-subset reporting pattern.

### 2.4 Benchmark-interface alignment is the highest-leverage, lowest-effort win
Rasouli's group ships a common data interface + action-prediction benchmark (PCPA, SF-GRU, …) that
reads both JAAD and PIE through a unified per-pedestrian sequence dict:
`{bbox, occlusion, action/behavior, intention, attributes, ego, image_paths}`. If our Step-3 output
mirrors those keys, existing SOTA baselines run on our data with near-zero glue — which turns the
Step-4 "reference baseline on GOLD as a label sanity check" into "run PCPA off-the-shelf."

**Proposal (to debate):** treat the benchmark dict as a *target output shape* for Step 3. **Open
question:** do we emit their exact schema, or our own schema + an adapter? (Adapter keeps our schema
clean but is one more thing to maintain.)

---

## 3. Field-by-field: what we can auto-derive, what we can't

### Auto-derivable now (no new models, from the tracked trajectory)
| field | how | value |
|-------|-----|-------|
| `action` (walking/standing) | speed threshold on the smoothed track | high — core model input in JAAD/PIE |
| `motion_direction` | velocity heading relative to ego | high — lateral vs longitudinal is a top crossing cue |
| `occlusion` 0/1/2 | map our per-frame signal: observed→`0`, low `num_lidar_points`/partial→`1`, coasted/bridged→`2` | direct match to their scheme; their models filter on it |
| `crossing` 1/0/−1 | from our corridor action; `−1` ≈ our `undetermined`/far `ped` | direct |

### Needs a model / budget (park for later)
| field | why it's hard | note |
|-------|---------------|------|
| `look` (looking/not-looking) | needs head-pose / gaze estimation | **JAAD/PIE's single strongest intent cue** — highest-value future add, fits the pipeline's reserved pose budget |
| `age`, `gender`, `gesture` | manual in JAAD/PIE; classifier + noise | low priority |
| appearance 24-dim | manual | `umbrella`/`phone`/`stroller` are real distraction cues a VLM could tag on the ~5% edge-case budget |

### Probably not obtainable cheaply from ZOD
`num_lanes`, `signalized` (C/S/CS), `intersection` (midblock/T/four-way), `traffic_direction` (OW/TW)
— need map / scene understanding ZOD doesn't hand us directly. **Do not fabricate these.** Open
question: does any ZOD road/scene metadata cover a subset?

### Where we are *richer* than JAAD/PIE
Ego context: we can emit PIE's per-frame fields (`OBD_speed`, `heading_angle`, `acceleration`) **plus
yaw-rate** (already computed — it's what curves our swept-path corridor), and we have LiDAR/radar
depth JAAD/PIE lack entirely. **Proposal:** name the ego fields exactly as PIE does for drop-in
comparability. Cheap, pure upside.

---

## 4. Summary of open decisions (for the supervisor)

1. **`agent_class`** — adopt pedestrian/ped/people? Promotion/demotion rules? Crowd → `people` trigger?
2. **Naming** — rename Step-3 label to *crossing-prediction*; reserve "intention" for the PIE sense?
   Attempt an intention proxy (pose/VLM) at all?
3. **Union tie-in** — the 2.4% vs 9.1% imbalance as empirical input to corridor-vs-union.
4. **Benchmark shape** — emit the PCPA/SF-GRU dict directly, or our schema + adapter?
5. **Which auto-derived fields to commit to** (action, motion_direction, occlusion, ego names) — likely
   yes, but confirm before touching the schema.
6. **`look`/pose** — is this worth prioritising given it's the strongest single cue?

Nothing above is implemented. Revisit `configs/dataset_schema_v0.2.yaml` only once these are settled.

---

## 5. Manual curation labeling rules (edge cases) — operational

These are the acting rules for the human anchor pass over the curated batch
(`data/processed/review/curation_worksheet.csv`), added 2026-07-27. They resolve the recurring
edge cases against the JAAD/PIE `crossing` scheme in §1 / §3 (`1` cross / `0` not / `−1` irrelevant).
Do keep/drop/merge **first**; label crossing only on the tracks that survive (a dropped or merged
fragment carries no crossing — its crossing lives with the primary it folds into).

| case | verdict | `crossed_yes_no` | `crossing_start` | why (JAAD/PIE mapping) |
|------|---------|------------------|------------------|------------------------|
| **Too far / too small to tell** | keep or drop | `undetermined` | — | JAAD `crossing = −1` (irrelevant / not observable). Never force a `no` — that poisons the negative class with peds we never actually saw. Most fail the Step-3 50 m ego gate anyway. |
| **Cyclist** (or scooter/wheelchair — not a walking pedestrian) | `drop`, note `cyclist` | — | — | JAAD annotates pedestrians only; cyclists get no crossing behavior. YOLO's `person` class swept it into SILVER. Exclude to match JAAD. |
| **Already mid-crossing at first sight** (e.g. we join the scene as the ego turns) | keep | `yes` | `pre-obs` (+ flag no-onset) | Action = they crossed. But onset happened off-camera, so there is no `t_c` to predict → JAAD/PIE drop already-crossing windows from the prediction set. The `pre-obs`/no-onset flag keeps it out of the positive **intent** windows in Step 3. |

**`crossing_start` convention.** Use the **same clock as the video overlay**: signed seconds relative
to the **keyframe** (`0` = keyframe), which is exactly how the overlay computes `t_c`
(`crossing_onset − keyframe`). So a ped stepping onto the road half a second before the keyframe is
`−0.5`; 3.5 s after is `+3.5`. This makes "human − machine" the direct t_c error. You do **not** need
frame precision — the overlay flashes a red border at the machine's onset frame; write `matches` when
it lands right, and only give your own signed-seconds value when it's clearly early/late/wrong.

## Sources (verified 2026-07-02)
- JAAD dataset repo — https://github.com/ykotseruba/JAAD
- PIE annotations README — https://github.com/aras62/PIE/blob/master/annotations/README.md
- PIE dataset page (York NVISION) — https://data.nvision2.eecs.yorku.ca/PIE_dataset/
- Rasouli et al., *PIE: A Large-Scale Dataset and Models for Pedestrian Intention Estimation and
  Trajectory Prediction*, ICCV 2019.
- *Coupling Intent and Action for Pedestrian Crossing Behavior Prediction* — arXiv:2105.04133.
