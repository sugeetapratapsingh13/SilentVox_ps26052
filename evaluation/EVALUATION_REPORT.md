# M3-P5 Evaluation Report

**Role:** Independent evaluator. This member measures pipelines built by
other members (classical/neural speech enhancement, ANC) and does not
build or modify those models. Where this report references M1's ANC
algorithm, it reports findings only — see the boundary note in Part 2.

**Status:** This document is the report template and methodology
record, wired to the actual metric code in `evaluation/`. The CSVs in
`results/` currently ship as header-only templates because no audio
corpus or model outputs were provided alongside this task. Run
`evaluation/evaluate.py` against real data (see Part 1–3 below) to
populate them, then fill in the "Findings" subsections here.

---

## Part 1 — Quantitative Evaluation

**Pipeline under test:**

```
Noisy Speech → Classical Enhancement → Neural Enhancement
```

**Metrics** (computed per file, per stage, against a clean reference):

| Metric | Script | Notes |
|---|---|---|
| STOI / ESTOI | `evaluation/stoi.py` | Intelligibility, [0,1]-ish, higher better |
| SI-SDR | `evaluation/sisdr.py` | Scale-invariant, dB, higher better |
| PESQ | `evaluation/pesq.py` | Only valid for narrowband/wideband speech; auto-resampled to 16k if the source rate isn't 8k/16k |
| SNR improvement | `evaluation/sisdr.py` (`snr_improvement`) | dB gained vs. the noisy baseline |
| Speech distortion | `evaluation/sisdr.py` (`speech_distortion_index`) | Fraction of output energy not explained by a scaled copy of clean speech — isolates speech damage from residual noise |
| Residual noise | `evaluation/sisdr.py` (`residual_noise_energy_ratio`) | Fraction of original noise energy still present |
| Latency, CPU, RAM | `evaluate.py` (`_profile_call`) | Live-timed if `--classical-fn`/`--neural-fn` given; otherwise left blank for you to fill from your own profiling run |
| Model size | `evaluate.py` (`_model_size_bytes`) | File size of `--classical-model` / `--neural-model` |

**How to run:**

```bash
python evaluation/evaluate.py part1 \
  --clean data/clean --noisy data/noisy \
  --classical data/classical_out --neural data/neural_out \
  --classical-fn mypkg.classical:process --neural-fn mypkg.neural:process \
  --classical-model models/classical_config.json --neural-model models/neural.onnx
```

Directories are matched by filename — `data/clean/x.wav`, `data/noisy/x.wav`,
`data/classical_out/x.wav`, `data/neural_out/x.wav` must all exist for `x.wav`
to be scored.

Outputs: `STOI_RESULTS.csv`, `SISDR_RESULTS.csv`, `PESQ_RESULTS.csv`,
`SNR_RESULTS.csv` (also carries latency/CPU/RAM/model-size columns),
`SPEECH_DISTORTION.csv`.

**Findings:** *(fill in after running against real data)*
- Classical vs. neural STOI/SI-SDR/PESQ delta:
- Which stage removes more noise vs. which distorts speech more:
- Resource trade-off (neural model size/latency vs. quality gain):

---

## Part 2 — ANC + Speech Preservation (cross-member boundary with M1)

**Pipeline under test:**

```
Original Speech + Noise → ANC (M1) → Speech + Residual Noise
                                    → Speech Enhancement → Final Communication Audio
```

**Boundary rule:** this evaluator reads M1's ANC output audio and scores
it. It does not read, call, or edit M1's ANC implementation. Any finding
below is a report **to** M1, not a patch **of** M1's code.

**Question asked:** does ANC reduce environmental noise while damaging
speech? Concretely, per file:

- `anc_noise_reduction_db` = SNR improvement from noisy input to ANC output
- `stoi_delta` = STOI(clean, ANC output) − STOI(clean, noisy input)
- `sisdr_delta_db` = SI-SDR(clean, ANC output) − SI-SDR(clean, noisy input)
- `flag_anc_damages_speech` = True when noise went down (`anc_noise_reduction_db > 0`)
  **and** speech quality also went down (`stoi_delta < -0.02` or `sisdr_delta_db < -0.5`)

These thresholds are a starting default — tune them once you see the
real score distribution; they exist to separate "ANC did nothing" from
"ANC actively traded speech quality for noise suppression."

If `--enhanced-output` is supplied (the speech-enhancement stage applied
downstream of ANC), the same clean-referenced STOI/SI-SDR are also
computed on that final output, so you can see whether downstream
enhancement recovers what ANC damaged.

**How to run:**

```bash
python evaluation/evaluate.py part2 \
  --clean data/clean --noisy data/noisy \
  --anc-output data/anc_out_from_M1 \
  --enhanced-output data/final_comm_audio
```

Output: `ANC_SPEECH_PRESERVATION.csv`, plus a console `[FINDING — report
to M1]` line listing flagged files when the pattern above is detected.

**Findings:** *(fill in after running against M1's actual ANC output)*
- Files flagged:
- Noise reduction achieved vs. speech quality lost, in aggregate:
- Recommendation to M1 (reported, not implemented here):

---

## Part 3 — Communication Scenarios

**Scenarios tested** (per the source spec):

| Key | Description |
|---|---|
| `mechanical_noise` | nearby speech + continuous mechanical noise |
| `engine_wind` | speech + engine + wind |
| `radio_environmental` | radio/communication audio + environmental noise |

**Metrics:** intelligibility (STOI/ESTOI), distortion (speech distortion
index), speech quality (SI-SDR, and PESQ where appropriate — PESQ is
skipped for scenarios whose "clean" reference is coded/radio audio
rather than plain speech, since PESQ is only validated for narrowband/
wideband speech; pass `--skip-pesq-for radio_environmental` for that).

**Expected directory layout:**

```
scenarios_root/
  mechanical_noise/{clean,noisy,processed}/*.wav
  engine_wind/{clean,noisy,processed}/*.wav
  radio_environmental/{clean,noisy,processed}/*.wav
```

**How to run:**

```bash
python evaluation/evaluate.py part3 \
  --scenarios-root data/comm_scenarios \
  --skip-pesq-for radio_environmental
```

Output: `COMMUNICATION_RESULTS.csv`.

**Findings:** *(fill in after running against real scenario recordings)*
- Best/worst scenario for intelligibility:
- Any scenario where noise reduction and intelligibility diverge:

---

## Running everything at once

```bash
python evaluation/evaluate.py all \
  --clean data/clean --noisy data/noisy --classical data/classical_out --neural data/neural_out \
  --anc-output data/anc_out_from_M1 --enhanced-output data/final_comm_audio \
  --scenarios-root data/comm_scenarios --skip-pesq-for radio_environmental \
  --results-dir results
```

## Known limitations / assumptions

- Length mismatches between paired files are handled by truncating to
  the shorter signal — no cross-correlation re-alignment. If any stage
  introduces significant delay (common with ANC), align signals before
  scoring or the deltas will be misleading.
- Live latency/CPU/RAM profiling (`--classical-fn`/`--neural-fn`) times
  the given Python callable in-process; it will not capture cost that
  happens outside the Python process (e.g. a separate inference server
  or embedded runtime). For those, supply a precomputed profiling log
  and merge it into `SNR_RESULTS.csv` manually, or extend `evaluate.py`.
- PESQ is gated to speech-appropriate inputs per the source's "where
  appropriate" instruction; it is intentionally left blank for
  non-speech or unscorable segments rather than reporting a
  meaningless number.
