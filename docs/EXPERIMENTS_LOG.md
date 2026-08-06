# Experiments Log

Dated record of what we measured and what we decided because of it. Chronological, append-only.
Raw reports live under `data/processed/reports/` (gitignored). Design rationale → `docs/PIPELINE.md`.

---

## Detector bring-up gates (2026-06-17/18) — drove the pivot away from 3D detectors

**Protocol.** 1,830 keyframe 3D GT boxes (358 seqs, 352 evaluated, 6 failed) for the 3D/frustum
gates; 2,159 GT 2D boxes for the 2D gate (includes 296 "2D-only" peds). 3D match = BEV centre, 2m
gate; 2D match = IoU ≥ 0.5; conf 0.1. "Compensated" = LiDAR motion-compensated to the camera keyframe
instant, which removes a ~5cm median / ~19cm p90 bias from the +37ms scan offset.

| gate | source | recall | precision | notes |
|---|---|---:|---:|---|
| 3D — nuScenes PointPillars | OpenPCDet PP-MultiHead, raw intensity, uncompensated | **0.485** | 0.085 | <40m 0.72–0.81; **>40m 0.24** |
| 3D — KITTI PointPillars | zhulf0804 `epoch_160`, intensity ×1/255, compensated | **0.114** | 0.229 | 0–10m 0.55 → **>40m 0.008** |
| 2D — YOLO11x | imgsz 2560, conf 0.1 | **0.629** | 0.645 | occl None 0.88 / Light 0.79 / Med 0.51 / Heavy 0.16 |
| **Frustum (2D→3D)** | YOLO box → projection → LiDAR depth slab, compensated | **0.585** | 0.533 | loc err median **0.147m**, p90 0.515; 0–10m 0.93 → >40m 0.44 |

Uncompensated frustum for reference: recall 0.574 / loc median 0.199m — compensation cut the median
error 26%.

**Conclusions.**
- Off-the-shelf PointPillars has a severe ZOD domain gap, worst beyond 40m where **~52% (949/1830)
  of GT lives**. KITTI weights (narrow front-FOV domain) are far worse than nuScenes. A transfer
  problem, not a tuning problem.
- **The frustum beats every off-the-shelf 3D detector** on recall, precision and range, with zero
  training, and gives ~15cm median 3D positions. Adopted as the Step-1 measurement front-end.
- 2D's limiter is occlusion (single-frame, recoverable by the linker) and small far peds. A better 2D
  detector lifts the frustum ceiling directly.

Also passed: **world-frame transform validation** — a known-static pole (`TrafficGuide`/`SnowMarker`)
with `location_3d`, run through the compensate-first chain, stayed pinned in world coordinates. This
validates pose interpolation + extrinsics for every design, not just this one.

Removed at git tag `experiments/3d-detectors`: `eval_detector_recall.py`, the PointPillars notebooks,
and the external OpenPCDet/PointPillars clones. Surviving: `bringup_*.py`,
`notebooks/02_bringup_gates.ipynb`.

---

## Boxfit cluster experiment (2026-06-26) — REJECTED, kept slab + anchor box

**Question.** Should a LiDAR-cluster centroid replace the nearest-depth slab as the tracking
measurement, and can a cluster supply box size/orientation? Tried: in-frustum points → ground removal
→ DBSCAN → nearest qualifying cluster → fitted oriented box. Gated on the keyframe GT box, 150 seqs,
384 cluster matches.

| centre err xy | slab | cluster |
|---|---:|---:|
| median | **0.146m** | 0.154m |
| mean | **0.232m** | 1.304m |
| p90 | **0.383m** | 1.17m |

Cluster box vs GT: 3D IoU median **0.098** (0% ≥0.5); height error median **−0.81m**; PCA yaw error
median **63°**.

**Conclusion — REJECTED.** The cluster ties the slab on median but has a heavy failure tail: for a
sparse or occluded pedestrian the body fails to form a qualifying cluster while a background blob in
the frustum cone does, so "nearest cluster" jumps onto it. The raw cluster box is unshippable —
sparse points underestimate height by ~0.8m, and a pedestrian's round cross-section makes PCA yaw
roughly random. Shipped box = **tracked (slab) centre + keyframe anchor size + velocity yaw**.

**The rejection is scoped to the case where the detector HAS a box.** When the detector MISSES
(occlusion), don't reach for clustering either: camera and LiDAR are co-mounted, so a hard occluder
hides the pedestrian from both — the cone holds the *occluder*. Handled by KF coast + RTS smooth.
Lesson: robustness comes from the motion prior and the gate, not from a better primitive.

Removed (in git history): `labeling/boxfit.py`, `bringup_boxfit_gate.py`, `viz_boxfit_diagnostic.py`.

---

## Step 2 corridor → BENCHED, crossing action = feet on ego road (2026-07-01 … 07-08)

The crossing action was originally *entry into the ego swept path*. Two findings retired it.

**The straight strip was wrong (07-01).** On seq 000041 (a 90° junction turn) it flagged 2/2
determined tracks as crossers — both people standing on the pavement. Replacing it with a curved
swept path (ped position projected onto the ego trajectory polyline) flagged 0/2. Dataset-wide the
corridor crossing count fell **185 → 38** (11.5% → 2.4% of determined); the ~147 lost were turn and
heading artifacts. The 38 survivors were clean (median min-lateral 0.05m, 32/38 with a real detection
at `t_c` and a left↔right side change).

**But 2.4% is too small a positive class**, and the broader `ego_road` rate was 9.1% — much closer to
JAAD/PIE. So (07-08) the action was **redefined as feet on the ego road** (`crosses_ego_road`).

Consequences: the corridor is no longer a label — its real value (metric range-to-crossing,
ego-relevance) is a *feature*. The computation is preserved dormant in `zodped.labeling.corridor`,
re-derivable as a Step-4 aux feature. Road-membership is FOV/range-limited because `ego_road` exists
only at the keyframe camera, but that IS the agreed definition and a camera model can emit it
per-frame. Re-ran full GOLD: 1,863 tracks, 1,602 determined, **146 `ego_road` crossings (9.11%)**.
Record schema `action/v0.2 → v0.3`.

---

## Tracker robustness: association in 3D vs 2D-first (2026-07-01, exploratory)

**Symptom.** The linker lifts boxes to 3D then associates in the world frame, so identity is decided
on the frustum-noisy *depth* axis while the reliable 2D box is discarded at the lift. Implausible
speed kicks (>4 m/s — a filter kick, not real motion) appear in **34/1545 (2%)** of GOLD tracks,
concentrated in crowds; at the kick the nearest *other* GOLD track is ≈0m away, i.e. an identity swap.

**A/B with a 2D-IoU pre-gate** (keep only candidates whose 2D box is IoU-consistent with the target's
last-seen box, before the existing 3D gate):

| seq | scene | peak speed | recall | verdict |
|---|---|---|---|---|
| 000903 | 5 peds, clean pass-by | 4.5 → 0.8 | 93→92% | **fixed** |
| 000953 | 3 peds | 4.3 → 0.9 | 64→41% | kick fixed, recall lost (stale last-box) |
| 001396 | 4 peds | 4.1 → 4.1 | 96→96% | unchanged — a frustum *depth-jump*, not a swap |
| 000603 | 34-ped crowd | 20→17 kicks | hurt | insufficient — overlapping boxes need appearance ReID |

**Reading.** Association is the right lever for pass-bys, but the pre-gate is a patch on a backwards
ordering. The correct architecture is **2D-first (associate → lift)** — PIPELINE "Direction & open
options" #3. Depth-jumps (measurement noise) and crowds (ReID) are separate problems.

---

## Committee gate: PV-LSTM on ZOD (2026-07-16 → 07-27)

Three passes. **The first two graded the model against geometry and were wrong about it** — read the
third.

**v1, graded vs geometry (07-16): looked like total failure.** Released JAAD-trained PV-LSTM ckpt
(validated on JAAD the same day: intention AUC 0.77, so the formatting was proven). Swept over all
1,863 GOLD tracks on a virtual 30 Hz timeline. Track AUC vs the geometric anchor **0.55/0.53/0.60** —
against 0.77 on JAAD. Event-locked profile flat-to-inverted. Not an aggregation artifact (max / mean /
top-5 / p90 all 0.42–0.56).

**Iteration (a), detector boxes (07-16).** Windows rebuilt from matched Step-0 YOLO boxes instead of
projected-3D boxes. The boxes ARE visibly different (width 19px vs 39px on the same ped) and the score
scale responded (max-F1 threshold 0.343 vs 0.101), but ranking barely moved: test AUC 0.587 vs 0.597.
Ego-speed correlation −0.05, so ego-motion streaming was acquitted as a confound. Cost: 423
mostly-far/occluded tracks lose all windows to detector gaps.

**v2, graded vs the HUMAN (07-27) — both earlier conclusions corrected.** Same scorer and window
machinery, but truth = the human `crossed_yes_no` in the curation worksheet (25 seqs, both tiers, kept
tracks only). AUC is the headline: ~20 positives is too few for a held-out split, and AUC needs no
threshold.

| box source | n | crossers | AUC vs HUMAN | mean score cross / non |
|---|---:|---:|---|---|
| projected3d | 83 | 21 | **0.678** | 0.227 / 0.164 |
| detector | 77 | 20 | **0.736** | 0.228 / 0.148 |

1. **PV-LSTM is NOT near-random on ZOD.** Against a human it ranks a real crosser above a non-crosser
   ~68–74% of the time. Most of the "gate v1 FAILED" signal was *geometry being wrong*, not the model.
2. **Detector boxes DO help** — 0.736 vs 0.678. Invisible while grading against geometry (0.587 vs
   0.597), obvious once the truth is correct.

**Caveat, still standing:** ~20 crossers, pooled, optimistic same-set threshold. An encouraging read
with a wide error bar, not a validated pass. The *direction* is trustworthy (consistent across both
box types, clean mean-score separation); the absolute number is not.

**Lesson worth carrying: never grade a model against a label you haven't validated.** Two conclusions
were wrong for eleven days because the yardstick was 30% wrong.

---

## Manual anchor batch: geometry vs human (2026-07-27)

First human-labeled anchor, over the frozen curated batch (`review/curation_worksheet.csv`), on the
35 GOLD kept tracks. The reciprocal of the committee gate: instead of "does a model match geometry",
it asks **"does geometry match a human"**.

Confusion (human, machine): no/no 19, yes/yes 6, no/yes 6, yes/no 4 → **agreement 71%**, machine
crossing precision ~50%, recall ~60%. Crossing rate among kept decided tracks 21% (inside the
JAAD/PIE band).

**It fails in both directions — the key result.** Six false positives: four sequences tagged
`crosser` contained *no* human-confirmed crossing at all. Four false negatives: two sequences tagged
`negative` each held two real crossers geometry never saw.

**Two root causes, partly separable:**
1. **FOV/range limit of the keyframe polygon** — `ego_road` exists only at the keyframe camera, so
   crossings far ahead or out of view are invisible. Frustum depth smear is the likely FP mechanism:
   a sidewalk walker's lifted feet land on the road polygon.
2. **Road-extent definition mismatch** (annotator testimony) — a substantial share of the
   disagreement is *definitional*, not perceptual. Geometry sometimes counts only the ego's own
   carriageway, and treats separated bike lanes inconsistently with the human. **Fixable by
   redefining the polygon extent, separately from and more cheaply than any model work.**

**SILVER contamination.** Of 51 SILVER tracks: 20 keep / 11 merge / 20 drop → only **39% survive as
their own new pedestrian** (GOLD survives 35/36). Cyclists and people-inside-cars (YOLO `person` false
positives) are a **SILVER-only** contaminant — GOLD is anchored on verified keyframe boxes, so they
cannot enter it.

---

## SILVER pre-pour cleanup + review queue (2026-07-28 … 30)

No third model added — PedGraph+ was benched (pose-only ceiling ~0.6 < PV-LSTM's 0.74). Committee =
**geometry + PV-LSTM + human**.

**Agreement rule (why two judges suffice).** On the 77 human-labeled tracks, where geometry and
PV-LSTM AGREE (62% of tracks) they match the human **92%** (44/48) — better than either alone
(geometry 82%, PV 70%). Ship rule: auto-accept agreements, hand-review disagreements. PV threshold
0.170 (max-F1 from the human gate).

**Cleanup (SILVER only — GOLD tracks are verified and one-per-person, skip both):**
- Free cut (`silver_cut.py`) drops **664 (7%)**. The three rules — JITTER flag, `quality_tier=bad`,
  `max_speed > 5 m/s` — select the *same* 664, so they are one signal, not three. Catches jitter and
  people-in-cars; does NOT catch cyclists or reflections. SHORT alone and SPARSE_LIFT are deliberately
  not drop signals (SHORT is the fragment signature → stitch it, don't drop it).
- Track stitch (`stitch_tracks.py`): **97% precision (36/37)** against the human worksheet's merges.
  Finalized tightened (`--base-residual-m 0.6 --residual-per-s 0.2`) → 1,190 groups, 1,422 fragments
  folded, max residual 0.94m, zero merges above 1m. 290 groups extend a GOLD primary (healing gold
  rather than creating new pedestrians).
- Net: 9,394 SILVER tracks → **~7,300 unique clean pedestrians**, ~4× GOLD.

**Resolved-pedestrian accounting (07-30; every column sums to its tier total).** "Peds" = fragments
folded, cut junk removed, curated tracks excluded.

| bucket | GOLD | SILVER |
|---|---:|---:|
| AGREE no-cross | 709 | 3,798 |
| AGREE cross | 64 | 77 |
| DISAGREE geo=cross / PV=no | 56 | 96 |
| DISAGREE geo=no / PV=cross | 532 | 1,484 |
| PV-blank, geo=cross | 4 | 46 |
| PV-blank, geo=no | 183 | 1,803 |
| undetermined (empty stub) | 261 | — |
| **resolved** | **1,809** (+54 curated) | **7,304** |

Both review piles are dominated by PV weakly firing "cross" where geometry says no (GOLD 532 / SILVER
1,484) — all low confidence (median ~0.24, none >0.5, PV-LSTM's conservative ceiling), so **not** the
review target. The tractable high-value review is the geometry=CROSS peds plus the agree-crossers.

**GOLD crosser review package:** all 146 GOLD crossers = 64 agree-cross + 56 geo=cross/PV=no +
4 geo=cross/PV-blank + 22 already curated. The 124 un-curated ones, plus 40 probe rows sampling the
highest-PV slice of the geo=no bucket, went to `review/gold_crosser_worksheet.csv` with one focus
clip each (`viz_render_video.py --layout split --ped`, 18s @ 10fps).

**Known limitation, accepted for v1.** That package catches geometry's false ALARMS but is nearly
blind to its MISSES — a missed crosser hides in the geo=no/PV=no slice nobody sampled.

---

## Two-reviewer crosser review, settled (2026-08-06)

The 164-row crosser worksheet was filled independently twice — once by the student, once by the
supervisor (71 of the 164 rows). **Inter-reviewer agreement: 61/71 = 86%.** That is the honest
ceiling on this label and belongs next to any model AUC we quote, because a model cannot beat the
yardstick's own noise. (Geometry, for scale, agreed with a human ~70%.)

| divergence | n | note |
|---|---:|---|
| different verdict | 10 | mostly "other road, not the ego road", and far-away peds |
| same verdict, crossing offsets >0.5s apart | 18 | looks settled, is not — gaps up to **9.0s** |

**The second row is the trap.** Both reviewers say "yes, crossing", so the row reads as agreed — but
`t_c` is what Step 3 anchors every prediction window to, so a contested crossing instant mislabels
window CONTENTS while the window COUNT stays healthy. Half a second is ~5 frames; past that, two
people are describing different events.

All 33 divergences were settled by re-watching together, and merged by hand back into the single
`gold_crosser_worksheet.csv`. One worksheet is the source of truth; no parallel tables.

**Merge result** (`02e_merge_human_labels.py` → `actions_verified/`), GOLD, 1,863 tracks:
217 human-verified (119 cross / 97 no-cross / 1 uncertain) + 1,646 geometry-kept. The human pass
**flipped 41 geometry crossers to no-cross and 15 no-cross to crossers**.

**Geometry vs human on all 217 reviewed:**

| geometry | human | n |
|---|---|---:|
| cross | cross | 104 |
| cross | **no** | **41** |
| no | **cross** | **15** |
| no | no | 56 |

**Geometry is wrong on 41 of the 145 crossings it declares — 28%.** Do NOT quote a recall number from
this table: the sample is biased (geometry-crossers plus a high-PV probe), so nobody looked at the
tracks where geometry and PV both said no. The true miss count is ≥15 and unknown.

---

## Step 3 rebuilt on human-verified labels (2026-08-06)

`03_assemble_samples.py --actions-dir data/processed/actions_verified` — the first Step-3 build in
which human verdicts actually reach the shipped samples.

| | geometry (Jul 15) | human-verified | Δ |
|---|---:|---:|---|
| samples | 1,087 | 1,043 | −44 |
| TTE-anchored (crosser) windows | 242 | 180 | **−62** |
| comparison windows | 845 | 863 | +18 |
| crossing ratio @1.0s | 0.081 | 0.063 | |
| crossing ratio @1.5s | 0.155 | 0.120 | |
| crossing ratio @2.0s | **0.223** | **0.173** | |

**The positive class shrank by a quarter, and that is correct** — geometry was calling 41
non-crossings crossings, so 22.3% was inflated by its false alarms. 17.3% is the honest number. It is
nonetheless BELOW the 20–30% target band, which is now a real question for the supervisor rather than
a satisfied constraint.

Two quality signals moved the right way, so this is not merely "fewer labels":
- `window_outside_track` skips **176 → 129 (−47)**: human `t_c` values sit inside the tracked span far
  more often than geometry's did. The anchors are better placed, not just rarer.
- `undetermined_action` 261 → **274 (+13)**: 12 `pre_obs` crossers plus 1 human-uncertain, all kept
  and flagged rather than forced.

**The frozen splits held.** Crossing ratio @2.0s is 0.171 train / 0.180 val / 0.175 test against a
0.173 corpus — the Step-4a stratification survived a labeling change it never saw, which is exactly
what freezing at sequence level was supposed to buy.

**Levers if the ratio must come back up:** only one survives measurement — chase geometry's misses
(the 40-row probe already found 15 crossers in the geometry=no bucket, so more exist in the slice
nobody sampled). **Not** the ego gate (a class-balance filter — see below) and **not** SILVER (it
dilutes to 2.6% — see the pour).

### The 50m ego gate is a class-balance filter — do NOT relax it (2026-08-06)

`distance_to_ego` cuts 458 windows, the biggest single filter, so relaxing it looked like the obvious
way to recover positives. Swept it (GOLD, verified labels, everything else held):

| max_distance_to_ego_m | crossers @2.0s | non-crossers | ratio @2.0s |
|---|---:|---:|---|
| **50 (current)** | 180 | 863 | **0.1726** |
| 75 | 186 | 1,104 | 0.1442 |
| 100 | 192 | 1,202 | 0.1377 |
| unlimited | 192 | 1,230 | 0.1350 |

**Removing it adds 12 crossers and 367 non-crossers** — the admitted region is ~3% crossing-dense
against the corpus's 17%, so the ratio falls by a fifth and the target band gets *further* away.

The mechanism is obvious in hindsight: crossing the ego road requires being near the ego road, and
the ego is on that road. Pedestrians beyond 50m are overwhelmingly non-crossers, or their crossing is
not observable from here. So the gate is not merely a range filter — it removes far-away negatives
faster than far-away positives, and that is most of what keeps the ratio where it is.

**Keep 50m.** If the band must be met, the honest levers are more positives (SILVER, geometry's
misses), not a wider window on the negatives.


---

## Committee re-measured on the 217 verified tracks — PV-LSTM does NOT hold up (2026-08-06)

The agreement rule that was going to label all of SILVER was justified by **77** tracks with ~20
crossers. Step 2e produced **217** human-verified tracks with 119 crossers, so the rule was
re-measured against them (`02b_committee_gate.py --truth verified --tier gold`, a new truth source on
the existing gate rather than a new script).

| | worksheet (77 tracks, 20 crossers) | verified (210 scored, 119 crossers) |
|---|---|---|
| PV-LSTM AUC vs human | **0.736** | **0.522** |
| mean score crosser / non | 0.228 / 0.148 | 0.237 / 0.232 |
| ship rule coverage (agree) | 61% | 44% |
| ship rule accuracy when agreeing | **91%** | **82%** |
| geometry alone | 81% | 75% |
| PV-LSTM alone | 70% | 53% |

**Not a code regression.** Re-running the old truth through the same code reproduces 0.736 / 61% /
91% exactly. The difference is entirely the population.

**Why the populations differ, and why the second one is the one that matters.** The worksheet batch
was a broad curated sample (26% crossers). The verified set is deliberately concentrated on the
decision boundary — every geometry-declared crosser plus a high-PV probe of the geometry=no bucket —
so it is 57% crossers and contains almost no easy far-away negatives. PV-LSTM's apparent 0.74 came
substantially from ranking easy irrelevant tracks below crossers. **On the hard cases where geometry
is actually uncertain, it scores 0.52 — a coin flip, with crosser and non-crosser mean scores
0.237 vs 0.232.** That is precisely the slice a tie-breaker exists to arbitrate.

Both samples are selected, so neither number is "PV-LSTM's true AUC on ZOD". But the decision only
depends on the boundary slice, and there the model adds nothing: the agreement rule buys 82% accuracy
on 44% of tracks, against geometry's 75% on 100% of them.

**Decision: do not ship the geometry+PV-LSTM committee for SILVER.** PV-LSTM stays useful for what it
demonstrably did — RANKING which tracks a human should watch, which is how the crosser review package
was built — and nothing more.

> **Follow-up (2026-08-06, same day): training does NOT overturn this.** The 0.52 above is a
> JAAD-trained checkpoint judged zero-shot, and retraining on ZOD lifts ranking to AUC 0.76 (see
> "Step 4b reference baseline"). But labeling needs precision at a threshold, not ranking: the
> trained model declares crossings at **0.29 precision** (0.46 at its best operating point) against
> **geometry's 0.72**. Relabeling with it would make SILVER's labels worse. The decision stands, now
> on stronger evidence.

**SILVER cannot currently be validated at all.** The 30 kept SILVER tracks carrying a human crossing
label are 28 no/no, with one geometry crosser and one human crosser that are different tracks. There
is effectively **one positive**, so that set can measure nothing about crossing accuracy.

**Consequence for the pour.** SILVER ships as *weak-labeled training bulk*: geometry labels, the
`pv_disputed` flag retained as a counted quantity, `is_in_gold_standard=false`, and the measured GOLD
error rate (28% false-positive on declared crossings) documented as the expected label noise. It must
never be used as an evaluation set — GOLD is the eval set, and GOLD is human-verified. This is what
the two-tier design was for; the committee was an attempt to do better than that, and it did not earn
its place.

**If SILVER labels must improve later**, the only honest basis is a *crosser-enriched random* sample
of SILVER labeled by a human — not more model votes, and not the current 30-track set.


---

## SILVER poured through Steps 2–3 (2026-08-06)

The cut + stitch manifests were manifests only — nothing honored them. Steps 2 and 3 now take
`--stitch`, which applies the free cut and folds fragments into their primary **in memory**
(`zodped.labeling.stitching.resolve_sequence_tracks`); no trajectory file is rewritten, so Step 1's
output stays the reproducible source. Measured first: keeping only the primary instead of merging
would have lost 12 of ~219 SILVER crossings and ~1,400 fragments' worth of frames, in the tier whose
whole purpose is volume — so merging is worth the loader.

**Resolution:** 11,257 track files → **9,247 pedestrians** (1,863 GOLD + 7,384 SILVER). 664 cut as
junk, ~1,346 fragments folded. 2,010 now-stale per-fragment action records removed so
`actions/` holds exactly one record per pedestrian.

**GOLD is never absorbed.** A first version of the loader dropped any non-primary member of a
gold-primary group, which silently deleted 2 GOLD pedestrians belonging to **GOLD–GOLD** stitch
proposals. Each GOLD track is anchored on its own verified ZOD keyframe box and its human label is
keyed to its id, so a GOLD member is now never absorbed whatever the proposal says. (Both such
proposals are almost certainly false merges; resolving them is a separate question.) No SILVER-primary
group contains a GOLD member, so the fix did not change any SILVER output.

**Result — and the finding that matters:**

| | samples | pedestrians | crossers @2.0s | ratio @2.0s |
|---|---:|---:|---:|---|
| GOLD | 1,043 | 929 | 180 | **0.173** |
| SILVER | 3,406 | 3,356 | 90 | **0.026** |
| combined | 4,449 | 4,285 | 270 | 0.061 |

**SILVER's crossing rate is 6.6× lower than GOLD's**, and it dilutes the corpus ratio from 17.3% to
6.1%. That is not a bug — it is what SILVER *is*. GOLD pedestrians were annotated by ZOD because a
human judged them relevant, and relevant overwhelmingly means near the ego road. SILVER pedestrians
are whatever the detector found: far, peripheral, on the pavement. Step 2 already showed it at track
level (SILVER 3.0% crossing vs GOLD 9.1%).

**Consequence: do not quote a combined crossing ratio.** It describes neither tier. GOLD is the
evaluation set at 17.3%; SILVER is training bulk at 2.6%, ~3.3× GOLD's sample volume for half its
positives. The index carries `is_in_gold_standard` and `label_confidence_tier` on every row, so the
two are always separable — report them separately.

**Known wart: scene context is tier-dependent.** `num_pedestrians_in_scene` and `is_key_pedestrian`
are derived from the tracks a run loaded, so a `--tier gold` run sees only GOLD neighbours (e.g. 9)
where the full pass sees the real population (37). Pouring SILVER changed those two fields on 1,027
of the 1,043 GOLD samples, and flipped `is_key_pedestrian` on 118 — **no label, count, or ratio moved**.
The shipped values are from the `--tier all` pass and are the accurate ones. Making the field
population-independent (always compute scene context over every track, whatever --tier processes) is
a small fix worth doing before anyone re-runs a single tier and gets different metadata.


---

## Step 4b reference baseline: PV-LSTM TRAINED on our data (2026-08-06)

`scripts/04b_train_baseline.py` — the PV-LSTM position+velocity architecture retrained at OUR
protocol (5-frame windows at ~10 Hz, not 16 at 30 Hz). Train on the frozen train split, select on
GOLD val, evaluate on **GOLD test only** (199 windows, 35 crossers) whatever the model trained on.
5 seeds per arm.

| arm | train windows | crossers | GOLD test AUC | AP |
|---|---:|---:|---|---|
| GOLD only | 740 | 127 | 0.763 ± 0.025 | 0.375 ± 0.029 |
| GOLD + SILVER | 3,071 | 186 | **0.800 ± 0.034** | 0.387 ± 0.064 |
| chance | | | 0.500 | 0.176 |

**Training fixes what zero-shot could not — the supervisor's hypothesis was correct.** The released
JAAD checkpoint scored **0.52** on our verified tracks; the same architecture trained on ZOD scores
**0.76–0.80**. So the earlier failure was a domain-gap result about a checkpoint, NOT a verdict on
the model or on our labels. AP is more than double chance (0.39 vs 0.18), so the labels carry real,
learnable signal — the Step-4b sanity check passes.

**Headline: the GOLD-only arm, AUC 0.763 ± 0.025.** It trains purely on human-verified labels, so it
is the number to quote; the SILVER arm is the ablation, not the result.

**Does SILVER earn its place? Not proven — but it does not poison.** Paired by seed, the SILVER arm
wins 3/5 with a mean gain of +0.036 AUC, and the per-seed deltas swing both ways (−0.012 to +0.114).
The seed spread measures initialisation only; with 35 test positives the sampling error is larger
still. So: **the "weak labels will poison the model" worry is answered — neither arm collapsed** — but
"SILVER improves the baseline" is not established by this evidence. Settling it needs a bigger
verified test set, not more seeds.

**Ranking is not labeling — the trained model still must not label.** AUC asks only whether crossers
outrank non-crossers, and it is forgiving at a 17.6% base rate where negatives outnumber positives
4.7:1. Converted to a decision, the trained model declares crossings at **0.29 precision** (0.46 at
its best operating point) against **geometry's 0.72**. So re-labelling GOLD or SILVER with these
weights would degrade the dataset, and relabelling GOLD would be circular besides — it learned those
labels. PV-LSTM's job stays ranking review candidates.

**Caveats to carry with the number.** 35 test positives is thin. Two humans agreed on only 86% of
these labels, so ~14% of the truth is itself contested — near-1.0 is not available, and a few points
between runs mean nothing. For scale, PV-LSTM scores 0.77 on JAAD's own test split.
