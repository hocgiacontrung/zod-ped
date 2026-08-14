# JAAD / PIE Alignment

What JAAD and PIE actually annotate, how it maps onto us, and the naming/taxonomy decisions still
open. Our novelty is the extra modalities and auto-labeling at scale; the *taxonomy* is a solved problem worth borrowing.

> **§1–4 are DRAFT FOR DEBATE. Nothing in them is adopted.** §5 (curation rules) IS operational.
> Specs verified 2026-07-02 from the repos, not memory — see Sources.

---

## 1. Verified annotation specs

**JAAD agent classes** — distinguished by relevance + resolvability, NOT data quality:

| class | id suffix | annotated | meaning |
|---|---|---|---|
| `pedestrian` | `b` | full behavior + attributes | near the road, interacts with the ego |
| `ped` | — | bbox only | bystander, far, no ego interaction |
| `people` | `p` | single group box | a group, not individually resolved |

`sample_type='all'` vs `'beh'` selects all pedestrians vs the behaviorally-annotated subset.

**PIE behavioral annotations** (per frame, per pedestrian) — `action`: walking/standing;
`gesture`: hand_ack / hand_yield / hand_rightofway / nod / other; `look`: looking / not-looking (at
the ego); `cross`: not-crossing / crossing / crossing-irrelevant.

**PIE pedestrian attributes** — `age` (child/adult/senior), `gender`, `num_lanes`, `signalized`
(n/a|C|S|CS), `traffic_direction` (OW|TW), `intersection` (midblock|T|T-right|T-left|four-way),
`crossing` (1|0|−1), `exp_start_point` / `critical_point` (first/last frame shown to raters),
**`intention_prob`** [0,1] aggregated from a human experiment, `crossing_point` (frame where crossing
starts — our `t_c`).

**Occlusion (both)** — pedestrians: 0 none, 1 partial (25–75%), 2 full (>75%). Other objects: 0
visible, 1 partial/full.

**PIE ego/OBD** (per frame) — GPS_speed, OBD_speed, heading_angle, latitude, longitude, pitch, roll,
yaw, acceleration, gyroscope.

**JAAD appearance** — 24-dim binary, manual: pose_{front,back,left,right}, clothes colour/length,
backpack, bag_*, cap, hood, sunglasses, umbrella, phone, baby, object, stroller/cart,
bicycle/motorcycle.

---

## 2. The findings that actually change our design

### 2.1 `agent_class` is orthogonal to GOLD/SILVER — we likely want both

GOLD/SILVER is a **provenance** axis (how a track is born). JAAD's class is a
**relevance/resolvability** axis. Independent; conflating them would be a mistake.

| JAAD class | our equivalent |
|---|---|
| `pedestrian` | `determined` track near the road → becomes an intent sample |
| `ped` | far `determined` tracks **+ our EMPTY/`undetermined` stubs** → kept as detections, excluded from intent samples (already our keep-and-flag behaviour) |
| `people` | **our crowd identity-swap failure mode** (the seq-000603 34-ped crowd) |

**The `people` row matters most.** JAAD's answer to dense crowds is *don't individually track them* —
one group box, no per-person behavior. That is a principled escape hatch for exactly the regime where
our tracker is weakest: we do **not** have to solve crowd ReID to ship a clean dataset.

**Open:** what geometric rule promotes to `pedestrian` vs demotes to `ped`? What density threshold
triggers `people`, detected from box overlap / tracker fragmentation / detection density? Does
`people` get an action at group level, or none?

### 2.2 "Intent" is the wrong *name* for what Step 3 computes

PIE keeps three distinct fields: `cross` (per-frame action), `crossing_point` (≈ our `t_c`), and
`intention_prob` — the last from a **human experiment** (subjects watch up to `critical_point` and
rate "will they cross?").

The uncomfortable part: **`intention_prob` ≠ "does the pedestrian cross within the horizon".**
Intention is a judgement about a person's *state*; a pedestrian can intend to cross and then not
(hesitation, gap rejection), or drift across with no prior intent. What Step 3 derives —
`t_c ∈ [window_end, window_end+h]` — is *future action within a horizon*, which PIE would call the
**crossing-prediction** label.

That's fine and standard (SF-GRU / PCPA predict the `cross` action, not `intention_prob`), and we
have no budget for a human experiment. But we should name it honestly.

**Open:** rename Step-3's output to *crossing-prediction*, keep "intent" as a loose alias, or attempt
a cheap intention proxy (pose "looking", or a VLM on an edge-case budget) at all?

### 2.3 Class imbalance

JAAD/PIE are heavily crossing-imbalanced, which is why JAAD ships both `all` and `beh` views. This
was the empirical argument that retired the corridor label: the corridor-alone rate (2.4%) was far
more extreme than theirs, while `ego_road` (9.1% at track level) lands close to JAAD/PIE. Regardless,
we should mirror JAAD's `all` + behavioral-subset reporting pattern.

### 2.4 Benchmark-interface alignment — highest leverage, lowest effort

Rasouli's group ships a common data interface + action-prediction benchmark (PCPA, SF-GRU) that reads
both JAAD and PIE through a unified per-pedestrian dict:
`{bbox, occlusion, action/behavior, intention, attributes, ego, image_paths}`. If Step-3 output
mirrors those keys, existing SOTA baselines run on our data with near-zero glue — which turns the
Step-4 "reference baseline as a label sanity check" into "run PCPA off-the-shelf".

**Open:** emit their exact schema, or ours + an adapter? (Adapter keeps our schema clean but is one
more thing to maintain.)

---

## 3. Field-by-field: what we can auto-derive

**Now, from the tracked trajectory, no new models:**

| field | how |
|---|---|
| `action` (walking/standing) | speed threshold on the smoothed track — core model input in JAAD/PIE |
| `motion_direction` | velocity heading relative to ego — lateral vs longitudinal is a top crossing cue |
| `occlusion` 0/1/2 | observed→0, low `num_lidar_points`/partial→1, coasted/bridged→2 |
| `crossing` 1/0/−1 | from our action; −1 ≈ our `undetermined` / far `ped` |

**Needs a model or budget:** `look` (looking/not-looking) needs head-pose/gaze — **JAAD/PIE's single
strongest intent cue**, so the highest-value future add. `age`/`gender`/`gesture` are manual there;
low priority. The 24-dim appearance set is manual, though `umbrella`/`phone`/`stroller` are real
distraction cues a VLM could tag on a small edge-case budget.

**Probably not obtainable from ZOD:** `num_lanes`, `signalized`, `intersection`, `traffic_direction`
— these need map/scene understanding ZOD doesn't hand us. **Do not fabricate them.** Open: does any
ZOD scene metadata cover a subset?

**Where we are richer:** we can emit PIE's per-frame ego fields *plus* yaw-rate, and we have
LiDAR/radar depth JAAD/PIE lack entirely. Naming the ego fields exactly as PIE does is cheap pure
upside for drop-in comparability.

---

## 4. Open decisions for the supervisor

1. **`agent_class`** — adopt pedestrian/ped/people? Promotion rules? Crowd trigger?
2. **Naming** — rename Step-3's label to *crossing-prediction*? Attempt an intention proxy at all?
3. **Benchmark shape** — emit the PCPA/SF-GRU dict directly, or ours + an adapter?
4. **Which auto-derived fields to commit to** (action, motion_direction, occlusion, PIE ego names) —
   likely yes, but confirm before touching the schema.
5. **`look`/pose** — worth prioritising, given it's the strongest single cue?

Nothing above is implemented. Revisit `configs/dataset_schema_v0.2.yaml` only once these are settled.

---

## 5. Manual curation rules (operational)

The acting rules for the human review passes. Do keep/drop/merge **first**; label crossing only on
the tracks that survive — a dropped or merged fragment carries no crossing, its crossing lives with
the primary it folds into.

| case | verdict | crossing | onset | why |
|---|---|---|---|---|
| **Too far / too small to tell** | keep or drop | `uncertain` | — | JAAD `crossing = −1` (not observable). Never force a `no` — that poisons the negative class with peds we never actually saw. Most fail the Step-3 50m gate anyway. |
| **Cyclist** (or scooter/wheelchair) | `drop`, note `cyclist` | — | — | JAAD annotates pedestrians only. YOLO's `person` class swept these into SILVER; exclude to match. |
| **Already mid-crossing at first sight** | keep | `yes` | `pre_obs` | The action happened, but onset was off-camera so there is no `t_c` to predict. JAAD/PIE drop already-crossing windows from the prediction set; the `pre_obs` sentinel keeps them out of the positive intent windows in Step 3. |

**Onset convention.** Signed seconds relative to the **keyframe** (`0` = keyframe) — the same clock
the video overlay uses, so "human − machine" is the direct `t_c` error. A ped stepping onto the road
half a second before the keyframe is `−0.5`; 3.5s after is `+3.5`. Frame precision is not needed: the
overlay flashes a red border at the machine's onset, so only give your own value when it's clearly
early, late, or wrong.

**Two reviewers.** Where a track is reviewed twice, agreement on the *verdict* is not enough — onsets
more than ~0.5s apart mean the two are describing different events, and `t_c` is what Step 3 anchors
to. Settle those before merging (see EXPERIMENTS_LOG 2026-08-06).

---

## Sources (verified 2026-07-02)
- JAAD repo — https://github.com/ykotseruba/JAAD
- PIE annotations README — https://github.com/aras62/PIE/blob/master/annotations/README.md
- PIE dataset page — https://data.nvision2.eecs.yorku.ca/PIE_dataset/
- Rasouli et al., *PIE: A Large-Scale Dataset and Models for Pedestrian Intention Estimation and
  Trajectory Prediction*, ICCV 2019.
- *Coupling Intent and Action for Pedestrian Crossing Behavior Prediction* — arXiv:2105.04133.
